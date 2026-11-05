#!/usr/bin/env python3
"""Gold reference for Laguna S 2.1: the official transformers implementation (laguna), run on the official NVFP4
checkpoint one decoder layer at a time, in float32.

The whole model does not fit in memory (experts: 113 B parameters, 450 GB in float32), so the script builds each
decoder layer of the official code, loads its weights (NVFP4 dequantized exactly, BF16 widened), runs it, and frees it.
The routed experts are swapped for a lazy equivalent with the same math that loads only the experts the router picks.
Masks (causal, and the 512-position sliding window) and the per-layer-type RoPE (YaRN on the full layers) are built
with the official mask functions and rotary module.

    tvenv/bin/python -I tests/ref_laguna.py --src models/Laguna-S-2.1-NVFP4 --tokens 1,2,3 --layers 48 --out ref.npy \
        [--dump-hidden hidden.f32] [--last N]

Outputs: logits [T or last N, vocab] float32 (.npy); with --dump-hidden, the residual stream after every layer,
float32 [L][T][3072].
"""
import argparse
import json
import os
import struct
import sys
import time

import numpy as np
import torch
from torch import nn

from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.models.laguna import modeling_laguna as M
from transformers.models.laguna.configuration_laguna import LagunaConfig

P = "model."
E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)


class ST:
    """Sharded safetensors reader over a (possibly still downloading) tools/hfget.py mirror."""

    DT = {"BF16": np.uint16, "F32": np.float32, "F8_E4M3": np.uint8, "U8": np.uint8, "I64": np.int64, "F16": np.float16}

    def __init__(self, d):
        self.d = d
        self.map = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self.hdr = {}
        self.mm = {}

    def _h(self, f):
        if f not in self.hdr:
            with open(os.path.join(self.d, f), "rb") as fp:
                n = struct.unpack("<Q", fp.read(8))[0]
                self.hdr[f] = (8 + n, json.loads(fp.read(n)))
        return self.hdr[f]

    def _covered(self, f, lo, hi):
        jp = os.path.join(self.d, ".hfget.done")
        if not os.path.exists(jp):
            return True
        iv = sorted((int(a), int(b)) for fn, a, b in (l.split() for l in open(jp) if l.strip()) if fn == f)
        pos = lo
        for a, b in iv:
            if a <= pos < b:
                pos = b
            if pos >= hi:
                return True
        return pos >= hi

    def has(self, name):
        return name in self.map

    def raw(self, name, wait=True):
        f = self.map[name]
        base, h = self._h(f)
        info = h[name]
        s, e = info["data_offsets"]
        waited = False
        while wait and not self._covered(f, base + s, base + e):
            if not waited:
                print(f"waiting for {name} to download...", file=sys.stderr)
                waited = True
            time.sleep(20)
        if f not in self.mm:
            self.mm[f] = np.memmap(os.path.join(self.d, f), dtype=np.uint8, mode="r")
        a = self.mm[f][base + s:base + e]
        return np.frombuffer(a, dtype=self.DT[info["dtype"]]).reshape(info["shape"]), info["dtype"]

    def tensor(self, name):
        """float32 torch tensor. `X.weight` of a compressed-tensors NVFP4 layer (X.weight_packed: two e2m1 codes per
        byte, low nibble first; X.weight_scale: e4m3 per 16 inputs; X.weight_global_scale) is dequantized as
        e2m1 * scale / global_scale."""
        if not self.has(name) and name.endswith(".weight") and self.has(name + "_packed"):
            base = name[: -len("weight")]
            codes = torch.from_numpy(self.raw(name + "_packed")[0].copy())
            q = torch.stack([codes & 15, codes >> 4], -1).reshape(codes.shape[0], codes.shape[1] * 2).long()
            sc = torch.from_numpy(self.raw(base + "weight_scale")[0].copy()).view(torch.float8_e4m3fn).float()
            gs = self.tensor(base + "weight_global_scale").reshape(())
            return E2M1[q] * sc.repeat_interleave(16, 1) / gs
        a, dt = self.raw(name)
        if dt == "BF16":
            return torch.from_numpy((a.astype(np.uint32) << 16).view(np.float32).copy())
        return torch.from_numpy(a.astype(np.float32) if a.dtype != np.int64 else a.copy())


class LazyExperts(nn.Module):
    """LagunaExperts whose weights are read (and dequantized) only for the experts a batch hits. The forward is the
    official one line by line; only the weight lookup differs."""

    def __init__(self, st, base, num_experts, act_fn):
        super().__init__()
        self.st, self.base, self.num_experts, self.act_fn = st, base, num_experts, act_fn

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts + 1)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            e = int(expert_idx)
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate_up = torch.cat([self.st.tensor(f"{self.base}{e}.gate_proj.weight"), self.st.tensor(f"{self.base}{e}.up_proj.weight")], 0)
            gate, up = nn.functional.linear(current_state, gate_up).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, self.st.tensor(f"{self.base}{e}.down_proj.weight"))
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states


# checkpoint names of the official modules' parameters that differ from the module paths
RENAME = {"mlp.gate.e_score_correction_bias": "mlp.experts.e_score_correction_bias"}


def load_weights(st, mod, prefix):
    sd = mod.state_dict()
    new = {}
    for name in sd:
        key = prefix + RENAME.get(name, name).replace("shared_experts.", "shared_expert.")
        t = st.tensor(key)
        if t.shape != sd[name].shape:
            t = t.reshape(sd[name].shape)
        new[name] = t
    res = mod.load_state_dict(new, strict=False)
    if res.missing_keys:
        raise KeyError(f"unloaded parameters: {res.missing_keys[:5]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--out", required=True)
    ap.add_argument("--last", type=int, default=0)
    ap.add_argument("--dump-hidden", default=None)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    st = ST(a.src)
    cfg = json.load(open(os.path.join(a.src, "config.json")))
    cfg.pop("quantization_config", None)
    config = LagunaConfig(**cfg)
    config._attn_implementation = "eager"
    tk = open(a.tokens[1:]).read() if a.tokens.startswith("@") else a.tokens
    ids = torch.tensor([[int(x) for x in tk.split(",") if x.strip()]])
    T = ids.shape[1]

    emb = st.tensor(P + "embed_tokens.weight")
    h = emb[ids[0]].unsqueeze(0)
    del emb
    pos = torch.arange(T).unsqueeze(0)
    rotary = M.LagunaRotaryEmbedding(config)
    mk = {"config": config, "inputs_embeds": h, "attention_mask": None, "past_key_values": None, "position_ids": pos}
    masks = {"full_attention": create_causal_mask(**mk, allow_is_causal_skip=False) if "allow_is_causal_skip" in
             create_causal_mask.__code__.co_varnames else create_causal_mask(**mk),
             "sliding_attention": create_sliding_window_causal_mask(**mk)}
    for k, v in masks.items():
        if v is not None and v.dtype == torch.bool:  # boolean masks (True = attend) as additive float masks
            masks[k] = torch.zeros(v.shape).masked_fill(~v, float("-inf"))
    pe = {t: rotary(h, pos, t) for t in ("full_attention", "sliding_attention")}
    if a.dump_hidden:
        open(a.dump_hidden, "wb").close()
    for l in range(a.layers):
        t0 = time.time()
        with torch.device("meta"):
            layer = M.LagunaDecoderLayer(config, l)
        if config.mlp_layer_types[l] == "sparse":
            layer.mlp.experts = LazyExperts(st, f"{P}layers.{l}.mlp.experts.", config.num_experts, M.ACT2FN[config.hidden_act])
        layer = layer.to_empty(device="cpu")
        load_weights(st, layer, f"{P}layers.{l}.")
        layer.eval()
        kind = config.layer_types[l]
        h = layer(h, attention_mask=masks[kind], position_ids=pos, position_embeddings=pe[kind])
        del layer
        print(f"layer {l} ({kind}) done in {time.time() - t0:.1f}s, |x| = {h[0].norm(dim=-1).mean():.2f}", file=sys.stderr)
        if a.dump_hidden:
            with open(a.dump_hidden, "ab") as fh:
                fh.write(h[0].float().numpy().tobytes())
    w = st.tensor(P + "norm.weight")
    x = h[0].float()
    hn = w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + config.rms_norm_eps))
    out = hn[-a.last:] if a.last else hn
    logits = out @ st.tensor("lm_head.weight").T
    np.save(a.out, logits.float().numpy())
    print("argmax", logits.argmax(-1).tolist())


if __name__ == "__main__":
    main()
