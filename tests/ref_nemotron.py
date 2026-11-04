#!/usr/bin/env python3
"""Gold reference for Nemotron 3.5 Super: the official transformers implementation (nemotron_h), run on the official
NVFP4 checkpoint one block at a time, in float32.

The whole model does not fit in memory (experts: 115 B parameters, 460 GB in float32), so the script builds each
block of the official code, loads its weights (NVFP4 and FP8 dequantized exactly, BF16 widened), runs it, and frees
it. The routed experts are swapped for a lazy equivalent with the same math that loads only the experts the router
picks. Everything else is the unmodified transformers code, with one setting: the Mamba-2 step size keeps the
model's dt_limit (0, inf) (config.time_step_limit, the training and vLLM behaviour) instead of the transformers
fallback's (time_step_min, inf).

The MTP head is not in transformers (it ignores the mtp.* weights); --mtp follows vLLM's NemotronHMTP:
eh_proj([enorm(embed(t+1)) | hnorm(norm_f(h_t))]), then an attention block and a MoE block of the official code,
then final_layernorm and the shared LM head.

    tvenv/bin/python -I tests/ref_nemotron.py --src models/Nemotron-3.5-Super-VL-NVFP4 --tokens 1,2,3 \
        --layers 88 --out ref.npy [--dump-hidden hidden.f32] [--last N] [--mtp mtp.npy]

Outputs: logits [T or last N, vocab] float32 (.npy); with --dump-hidden, the residual stream after every block,
float32 [L][T][4096]; with --mtp, the MTP logits of slots 0..T-2 (slot t sees token t+1).
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

from transformers.models.nemotron_h import modeling_nemotron_h as M
from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig

P = "language_model.backbone."
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
        """float32 torch tensor: FP8 times its per-tensor weight_scale; NVFP4 codes times their e4m3 block scales
        and weight_scale_2 (modelopt layout: two e2m1 codes per byte, low nibble first, one scale per 16)."""
        a, dt = self.raw(name)
        base = name[: -len("weight")] if name.endswith("weight") else None
        if dt == "BF16":
            return torch.from_numpy((a.astype(np.uint32) << 16).view(np.float32).copy())
        if dt == "F8_E4M3":
            t = torch.from_numpy(a.copy()).view(torch.float8_e4m3fn).float()
            if base and self.has(base + "weight_scale"):
                t = t * self.tensor(base + "weight_scale").reshape(())
            return t
        if dt == "U8":
            codes = torch.from_numpy(a.copy())
            q = torch.stack([codes & 15, codes >> 4], -1).reshape(a.shape[0], a.shape[1] * 2).long()
            sc = torch.from_numpy(self.raw(base + "weight_scale")[0].copy()).view(torch.float8_e4m3fn).float()
            s2 = self.tensor(base + "weight_scale_2").reshape(())
            return E2M1[q] * sc.repeat_interleave(16, 1) * s2
        return torch.from_numpy(a.astype(np.float32) if a.dtype != np.int64 else a.copy())


class LazyExperts(nn.Module):
    """NemotronHExperts whose weights are read (and NVFP4-dequantized) only for the experts a batch hits. The
    forward is the official one line by line; only the weight lookup differs."""

    def __init__(self, st, base, num_experts, act_fn):
        super().__init__()
        self.st, self.base, self.num_experts, self.act_fn = st, base, num_experts, act_fn

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states, dtype=top_k_weights.dtype)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts + 1)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero().squeeze(-1)
        for expert_idx in expert_hit:
            e = expert_idx.item()
            top_k_pos, token_idx = torch.where(expert_mask[e])
            if token_idx.numel() == 0:
                continue
            current_state = hidden_states[token_idx]
            current_hidden_states = nn.functional.linear(current_state, self.st.tensor(f"{self.base}{e}.up_proj.weight"))
            current_hidden_states = self.act_fn(current_hidden_states)
            current_hidden_states = nn.functional.linear(current_hidden_states, self.st.tensor(f"{self.base}{e}.down_proj.weight"))
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states.to(hidden_states.dtype)


def load_weights(st, mod, prefix):
    sd = mod.state_dict()
    new = {}
    for name in sd:
        key = prefix + name
        if name == "mixer.gate.e_score_correction_bias" or name.endswith("gate.e_score_correction_bias"):
            new[name] = st.tensor(key).float()
            continue
        if not st.has(key):
            raise KeyError(f"missing {key}")
        t = st.tensor(key)
        if t.shape != sd[name].shape:
            t = t.reshape(sd[name].shape)
        new[name] = t
    res = mod.load_state_dict(new, strict=False)
    bad = [k for k in res.missing_keys if ".experts." not in k]
    if bad:
        raise KeyError(f"unloaded parameters: {bad[:5]}")


def build_block(st, config, l, prefix, kind):
    config.layers_block_type[l] = kind
    with torch.device("meta"):
        blk = M.NemotronHBlock(config, l)
    if kind == "moe":
        blk.mixer.experts = LazyExperts(st, f"{prefix}mixer.experts.", config.n_routed_experts, M.ACT2FN[config.mlp_hidden_act])
    blk = blk.to_empty(device="cpu")
    load_weights(st, blk, prefix)
    if kind == "linear_attention":
        blk.mixer.time_step_limit = tuple(config.time_step_limit)  # the model's (0, inf), see the docstring
    return blk.eval()


def rmsnorm(x, w, eps):
    x = x.float()
    return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--layers", type=int, default=88)
    ap.add_argument("--out", required=True)
    ap.add_argument("--last", type=int, default=0)
    ap.add_argument("--dump-hidden", default=None)
    ap.add_argument("--mtp", default=None)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    st = ST(a.src)
    cfg = dict(json.load(open(os.path.join(a.src, "config.json")))["llm_config"])
    legacy = {"mamba": "linear_attention", "attention": "full_attention", "moe": "moe", "mlp": "mlp"}
    kinds = [legacy[k] for k in cfg["layers_block_type"]]
    cfg["layers_block_type"] = list(kinds)
    cfg["mtp_layers_block_type"] = [legacy[k] for k in cfg.get("mtp_layers_block_type", ["attention", "moe"])]
    if isinstance(cfg.get("time_step_limit"), list):
        cfg["time_step_limit"] = (0.0, float("inf"))
    config = NemotronHConfig(**cfg)
    config._attn_implementation = "eager"
    eps = config.layer_norm_epsilon
    tk = open(a.tokens[1:]).read() if a.tokens.startswith("@") else a.tokens
    ids = torch.tensor([[int(x) for x in tk.split(",") if x.strip()]])
    T = ids.shape[1]

    emb = st.tensor(P + "embeddings.weight")
    h = emb[ids[0]].unsqueeze(0)
    mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1).view(1, 1, T, T)
    if a.dump_hidden:
        open(a.dump_hidden, "wb").close()
    for l in range(a.layers):
        t0 = time.time()
        blk = build_block(st, config, l, f"{P}layers.{l}.", kinds[l])
        h = blk(h, attention_mask=mask if kinds[l] == "full_attention" else None)
        del blk
        print(f"block {l} ({kinds[l]}) done in {time.time() - t0:.1f}s, |x| = {h[0].norm(dim=-1).mean():.2f}", file=sys.stderr)
        if a.dump_hidden:
            with open(a.dump_hidden, "ab") as fh:
                fh.write(h[0].float().numpy().tobytes())
    hn = rmsnorm(h[0], st.tensor(P + "norm_f.weight"), eps)
    lm = st.tensor("language_model.lm_head.weight")
    out = hn[-a.last:] if a.last else hn
    logits = out @ lm.T
    np.save(a.out, logits.float().numpy())
    print("argmax", logits.argmax(-1).tolist())
    if a.mtp and T > 1:
        Q = "language_model.mtp.layers."
        x = torch.cat([rmsnorm(emb[ids[0, 1:]], st.tensor(Q + "0.enorm.weight"), eps),
                       rmsnorm(hn[:-1], st.tensor(Q + "0.hnorm.weight"), eps)], -1)
        x = (x @ st.tensor(Q + "0.eh_proj.weight").T).unsqueeze(0)
        n = T - 1
        mmask = torch.triu(torch.full((n, n), float("-inf")), diagonal=1).view(1, 1, n, n)
        for i, kind in enumerate(config.mtp_layers_block_type):
            blk = build_block(st, config, 0, f"{Q}{i}.", kind)
            x = blk(x, attention_mask=mmask if kind == "full_attention" else None)
        x = rmsnorm(x[0], st.tensor(Q + "1.final_layernorm.weight"), eps)
        ml = x @ lm.T
        np.save(a.mtp, ml.float().numpy())
        print("mtp argmax", ml.argmax(-1).tolist())


if __name__ == "__main__":
    main()
