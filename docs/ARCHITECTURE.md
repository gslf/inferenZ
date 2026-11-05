# Architecture

[README](../README.md)

inferenZ is a single-model engine. The model's shapes are compile-time constants (`src/model.h`), checked against the GGUF at load. Kernels, memory layout and scheduling are written for this model and for CPUs. The model itself is described in [MODEL.md](MODEL.md).

## Source map

| File | Role |
|---|---|
| `main.c` | CLI: `serve`, `run`, `bench`, `convert`, `info`, `dump` |
| `convert.c`, `safetensors.c` | official NVFP4 checkpoint (compressed-tensors) → inferenZ GGUF; streams while the download is still running (reads the `hfget` journal) |
| `quant.c` | Q4N (NVFP4), Q8G and LUT kernels, Hadamard rotation, quantizer; matvec, batched, int8 GEMM and F16 GEMM kernels ([QUANTIZATION.md](QUANTIZATION.md)) |
| `model_fwd.c` | forward pass: gated grouped-query attention (global with YaRN RoPE, or a 512-token sliding window), dense MLP, MoE, LM head |
| `expert_cache.c` | io_uring + O_DIRECT expert cache with LRFU eviction, one per expert format ([STREAMING.md](STREAMING.md)) |
| `engine.c` | loading, memory budget, attention caches and rewinds, generation loop |
| `threadpool.c` | hybrid-CPU thread pool |
| `diag.c` | stage / phase / roofline counters, Perfetto traces, Prometheus ([DIAGNOSTICS.md](DIAGNOSTICS.md)) |
| `tokenizer.c` | Laguna pre-tokenizer (a cut before each newline run, then the split regex) + byte-level BPE (also the o200k, Qwen and Tekken splits) |
| `chat.c` | C port of the chat template; reasoning / content / tool-call stream parser |
| `server.c`, `http.c` | OpenAI-compatible API, SSE, web UI ([API.md](API.md)) |
| `gguf.c`, `json.c`, `sampler.c`, `util.c` | infrastructure |

## Memory tiers

What lives where, from the fastest tier to the slowest:

| Tier | Data | How it is held |
|---|---|---|
| RAM | dense weights, read for every token (3.7 GB) | one anonymous arena, loaded once, never paged out |
| RAM | expert caches: the experts in use | two huge-page arenas (NVFP4 experts, 8-bit experts) with LRFU eviction, as many experts as the memory budget allows |
| RAM | attention caches | K/V in f16: the 12 global layers by position, the 36 sliding layers in rings of 4096 positions; pages are touched only as the context grows |
| SSD | the other experts | one 4 KiB-aligned blob per expert, one `O_DIRECT` read per miss |
| SSD | embeddings | read with `pread` per token, through the page cache |

## One decode step

1. **Embedding.** Read the token's row.
2. **For each of the 48 layers:**
   1. **Attention.** RMSNorm; q/k/v and gate projections; per-head RMSNorm of q and k; RoPE (YaRN on the first 64 dims of each head in global layers, plain on all 128 in sliding ones); K/V appended to the cache; grouped-query softmax attention over every earlier position (global) or the last 512 (sliding); each head scaled by softplus of its gate; o projection; residual.
   2. **MLP.** RMSNorm; layer 0 runs a dense SwiGLU MLP. Layers 1-47: the router picks 10 of 256 experts (sigmoid scores, a bias that only shifts the choice, weights normalized and scaled by 2.5); resident experts run first, missing ones as they arrive from the SSD; the shared expert adds its output; residual.
   3. **Lookahead.** The next layer's MLP norm and router run on the updated residual, and the experts they predict are prefetched while its attention runs.
3. **Head.** The final RMSNorm and the LM head give the logits; a token is sampled.

## One prefill chunk (up to 2048 tokens)

Same layer structure, batched. Quantized projections use the int8 VNNI GEMM. Tokens are grouped by expert, so every needed expert is read once per chunk and applied to all of its tokens.

## Sequence state

Every cache is positional, so the engine can rewind to any earlier position whose sliding window the rings still hold (at most 3584 positions back). A chat client that resends the conversation reuses the whole common prefix of the previous request.


## Threading

`tp_for` hands out chunks of rows through an atomic cursor, so fast P-cores simply take more chunks than E-cores and there are no stragglers.
- **Placement.** Workers are pinned, one per physical P or E core. SMT siblings and the low-power E-cores are left to the HTTP server and I/O completion.
- **Idle waiting.** Idle workers wait with `UMONITOR/UMWAIT` (WAITPKG) before falling back to a futex. On a laptop this keeps the turbo budget for the cores doing work.

## Correctness strategy

- **Reference.** `tests/ref_laguna.py` runs the official transformers implementation (`laguna`) on the official NVFP4 checkpoint, one layer at a time in float32. The routed experts are replaced by a lazy loader with the same math, so only the experts the router picks are loaded.
- **Layer by layer.** `INZ_DUMP_HIDDEN=file inz dump` writes the engine's residual after every layer. `tests/compare_hidden.py` compares it with the reference's `--dump-hidden`. A bug shows up as a jump at the layer that has it.
- **Logits.** `tests/compare_logits.py` gives KL, cosine, top-1 and top-5 agreement.
- **Byte-exact parity tests:**
  - tokenizer vs HF `tokenizers` (`tests/tokenizer_parity.py`);
  - chat template vs Jinja (`tests/chat_parity.py`).
- **Code quality.** `tests/code_eval.py` gives the served model C and JavaScript tasks with hidden tests. Its `--selftest` checks the tests against reference solutions.
