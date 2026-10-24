#!/usr/bin/env python3
"""Gold reference for Qwen3.8-Flash-Next: the official transformers implementation (qwen4_exp), run on the
official FP8 checkpoint one decoder layer at a time, in float32.

The whole model does not fit in memory (experts: 121 GB in FP8, 483 GB in float32; n-gram table: 51 GB), so
the script builds each decoder layer of the official code, loads its weights (FP8 dequantized with the 128x128
block scales), runs it, and frees it. Two modules are swapped for lazy equivalents with the same math: the
n-gram embedding reads only the rows it needs from the checkpoint shards, and the routed experts load only the
experts the router picks. Everything else is the unmodified transformers code.

    tvenv/bin/python -I tests/ref_qwen38.py --src models/Qwen3.8-Flash-Next-FP8 --tokens 248044,... \
        --layers 48 --out ref.npy [--dump-hidden hidden.f32] [--last N]

Outputs: logits [T or last N, vocab] float32 (.npy); with --dump-hidden, the 10240-wide residual stream after
every layer, float32 [L][T][10240].
"""
import argparse
import glob
import json
import os
import struct
import sys
import time

import numpy as np
import torch
from torch import nn

from transformers.masking_utils import create_causal_mask
from transformers.models.qwen4_exp import modeling_qwen4_exp as M
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

P = "model.language_model."


class ST:
    """Sharded safetensors reader over a (possibly still downloading) tools/hfget.py mirror."""

    DT = {"BF16": np.uint16, "F32": np.float32, "F8_E4M3": np.uint8, "I64": np.int64, "F16": np.float16}

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
        """float32 torch tensor; FP8 weights are dequantized with their weight_scale_inv blocks."""
        a, dt = self.raw(name)
        if dt == "BF16":
            t = torch.from_numpy((a.astype(np.uint32) << 16).view(np.float32).copy())
        elif dt == "F8_E4M3":
            t = torch.from_numpy(a.copy()).view(torch.float8_e4m3fn).float()
            sname = name + "_scale_inv"
            if self.has(sname):
                s = self.tensor(sname)
                r, c = t.shape
                br, bc = -(-r // s.shape[0]), -(-c // s.shape[1])  # 128 x 128 blocks
                t = t * s.repeat_interleave(br, 0)[:r].repeat_interleave(bc, 1)[:, :c]
        else:
            t = torch.from_numpy(a.astype(np.float32) if a.dtype != np.int64 else a.copy())
        return t


class LazyNGram(nn.Module):
    """Stand-in for nn.Embedding(320M, 160): rows come straight from the FP8 shards (contiguous split, dim 0)."""

    def __init__(self, st, prefix, nshards):
        super().__init__()
        self.st, self.prefix, self.nshards = st, prefix, nshards
        a, _ = st.raw(f"{prefix}.shard_0.weight", wait=False)
        self.rows, self.dim = a.shape
        self.weight = torch.empty(0)  # keeps code that inspects .weight.device working
        # FP8Embedding: rows are multiplied by one per-tensor scale (BF16, ~2e-4)
        self.scale = float(st.tensor(f"{prefix}.weight_scale").reshape(-1)[0])

    def forward(self, ids):
        flat = ids.reshape(-1).tolist()
        out = torch.empty(len(flat), self.dim)
        for i, r in enumerate(flat):
            s, k = divmod(int(r), self.rows)
            a, dt = self.st.raw(f"{self.prefix}.shard_{s}.weight")
            assert dt == "F8_E4M3"
            out[i] = torch.from_numpy(a[k].copy()).view(torch.float8_e4m3fn).float()
        return (out * self.scale).view(*ids.shape, self.dim)


class LazyExperts(nn.Module):
    """Qwen4ExpTextExperts whose weights are read (and FP8-dequantized) only for the experts a batch hits.
    The forward is the official one line by line; only the weight lookup differs."""

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
            gate_up = torch.cat([self.st.tensor(f"{self.base}{e}.gate_proj.weight"),
                                 self.st.tensor(f"{self.base}{e}.up_proj.weight")], 0)
            gate, up = nn.functional.linear(current_state, gate_up).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states,
                                                         self.st.tensor(f"{self.base}{e}.down_proj.weight"))
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states


def load_layer_weights(st, mod, prefix, n_experts):
    sd = mod.state_dict()
    new = {}
    for name in sd:
        if ".ple_embedding.ngram_embedding" in name:
            continue
        if name.endswith("mlp.experts.gate_up_proj"):
            base = prefix + name[: -len("gate_up_proj")]
            new[name] = torch.stack([torch.cat([st.tensor(f"{base}{e}.gate_proj.weight"),
                                                st.tensor(f"{base}{e}.up_proj.weight")], 0) for e in range(n_experts)])
            continue
        if name.endswith("mlp.experts.down_proj"):
            base = prefix + name[: -len("down_proj")]
            new[name] = torch.stack([st.tensor(f"{base}{e}.down_proj.weight") for e in range(n_experts)])
            continue
        key = prefix + name
        if not st.has(key):
            # buffers computed at init (rope, n-gram multipliers/offsets) need no checkpoint value
            if name.endswith(("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets", "inv_freq",
                              "original_inv_freq")):
                continue
            raise KeyError(f"missing {key}")
        t = st.tensor(key)
        if t.shape != sd[name].shape:
            t = t.reshape(sd[name].shape)
        new[name] = t
    missing = mod.load_state_dict(new, strict=False)
    bad = [k for k in missing.missing_keys if "ngram_embedding" not in k and not k.endswith(
        ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets", "inv_freq", "original_inv_freq"))]
    if bad:
        raise KeyError(f"unloaded parameters: {bad[:5]}")


def build(cls_fn):
    # construct on the meta device (no allocation), then materialize on CPU
    with torch.device("meta"):
        m = cls_fn()
    return m


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
    cfg = json.load(open(os.path.join(a.src, "config.json")))["text_config"]
    config = Qwen4ExpTextConfig(**cfg)
    config._attn_implementation = "eager"
    tk = open(a.tokens[1:]).read() if a.tokens.startswith("@") else a.tokens
    ids = torch.tensor([[int(x) for x in tk.split(",") if x.strip()]])
    T = ids.shape[1]

    emb = st.tensor(P + "embed_tokens.weight")
    inputs_embeds = emb[ids[0]].unsqueeze(0)
    del emb
    rotary = M.Qwen4ExpTextRotaryEmbedding(config)
    position_ids = torch.arange(T).view(1, 1, -1).expand(3, 1, -1)
    text_pos = torch.arange(T).view(1, -1)
    pe = rotary(inputs_embeds, position_ids)
    mask = create_causal_mask(config=config, inputs_embeds=inputs_embeds, attention_mask=None, past_key_values=None,
                              position_ids=text_pos, allow_is_causal_skip=False)
    h = inputs_embeds.repeat(1, 1, config.hc_count)
    if a.dump_hidden:
        open(a.dump_hidden, "wb").close()
    for l in range(a.layers):
        t0 = time.time()
        layer = build(lambda: M.Qwen4ExpTextDecoderLayer(config, l))
        if layer.ple is not None:
            # swap the 320M x 160 embedding for row-on-demand reads before anything is materialized
            layer.ple.ple_embedding.ngram_embedding = LazyNGram(
                st, f"{P}layers.{l}.ple.ple_embedding.ngram_embedding", config.split_ngram_parts)
        # routed experts are read on demand (a prompt of T tokens touches at most 10 T of the 512)
        layer.mlp.experts = LazyExperts(st, f"{P}layers.{l}.mlp.experts.", config.num_experts, M.ACT2FN[config.hidden_act])
        layer = layer.to_empty(device="cpu")
        if layer.ple is not None:
            # to_empty() wiped the hashing buffers: rebuild them (identical to the checkpoint's copies)
            ng = layer.ple.ple_embedding
            ng.layer_multipliers = M._build_layer_multipliers(ng.unigram_vocab_size, ng.ngram_size,
                                                              ng.ple_layer_index, ng.seed)
            ng.ngram_heads_vocab_sizes = torch.tensor(ng.head_vocab_sizes, dtype=torch.long)
            ng.ngram_heads_offsets = torch.tensor(ng.head_offsets, dtype=torch.long)
        load_layer_weights(st, layer, f"{P}layers.{l}.", config.num_experts)
        layer.eval()
        h = layer(h, position_embeddings=pe, attention_mask=mask, conv_mask=None, past_key_values=None,
                  ple_input_ids=ids)
        del layer
        norms = h[0].view(T, config.hc_count, -1).norm(dim=-1).mean(-1)
        print(f"layer {l} done in {time.time() - t0:.1f}s, |x| = {norms.round(decimals=2).tolist()}", file=sys.stderr)
        if a.dump_hidden:
            with open(a.dump_hidden, "ab") as fh:
                fh.write(h[0].float().numpy().tobytes())
    mixer = build(lambda: M.Qwen4ExpTextGatedResidual(config, use_combine=False)).to_empty(device="cpu")
    load_layer_weights(st, mixer, f"{P}hyper_connection_mixer.", 0)
    out = mixer(h)[0]
    if a.last:
        out = out[-a.last:]
    lm = st.tensor("lm_head.weight")
    logits = out @ lm.T
    np.save(a.out, logits.float().numpy())
    print("argmax", logits.argmax(-1).tolist())


if __name__ == "__main__":
    main()
