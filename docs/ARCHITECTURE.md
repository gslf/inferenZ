# Architecture

[README](../README.md)

inferenZ is a single-model engine. The model's shapes are compile-time constants (`src/model.h`), checked against the GGUF at load. Kernels, memory layout and scheduling are written for this model and for CPUs. The model itself is described in [MODEL.md](MODEL.md).

## Source map

| File | Role |
|---|---|
| `main.c` | CLI: `serve`, `run`, `bench`, `convert`, `info`, `dump` |
| `convert.c`, `safetensors.c` | official NVFP4 checkpoint → inferenZ GGUF; streams while the download is still running (reads the `hfget` journal) |
| `quant.c` | Q4N (NVFP4) and F8T (FP8) kernels, LUT quants, Hadamard rotation, quantizer; matvec, batched, int8 GEMM and F16 GEMM kernels ([QUANTIZATION.md](QUANTIZATION.md)) |
| `model_fwd.c` | forward pass: Mamba-2, attention, LatentMoE, MTP blocks, LM head |
| `expert_cache.c` | io_uring + O_DIRECT expert cache with LRFU eviction ([STREAMING.md](STREAMING.md)) |
| `engine.c` | loading, memory budget, sequence state and checkpoints, generation and speculative loops ([SPECULATIVE.md](SPECULATIVE.md)) |
| `threadpool.c` | hybrid-CPU thread pool |
| `diag.c` | stage / phase / roofline counters, Perfetto traces, Prometheus ([DIAGNOSTICS.md](DIAGNOSTICS.md)) |
| `tokenizer.c` | Tekken pre-tokenizer split + byte-level BPE (also the o200k and Qwen splits) |
| `chat.c` | C port of the chat template; reasoning / content / tool-call stream parser |
| `server.c`, `http.c` | OpenAI-compatible API, SSE, web UI ([API.md](API.md)) |
| `gguf.c`, `json.c`, `sampler.c`, `util.c` | infrastructure |

## Memory tiers

What lives where, from the fastest tier to the slowest:

| Tier | Data | How it is held |
|---|---|---|
| RAM | dense weights, read for every token (7.5 GB) | one anonymous arena, loaded once, never paged out |
| RAM | expert cache: the experts in use | huge-page arena with LRFU eviction, as many experts as the memory budget allows |
| RAM | recurrent state | the Mamba-2 states (168 MB, f32), their checkpoint and the speculative snapshots |
| RAM | attention caches | K/V in f16 for 9 attention blocks (8 + the MTP block); pages are touched only as the context grows |
| SSD | the other experts | one 4 KiB-aligned blob per expert, one `O_DIRECT` read per miss |
| SSD | embeddings | read with `pread` per token, through the page cache |

## One decode step

1. **Embedding.** Read the token's row.
2. **For each of the 88 blocks:** RMSNorm of the residual, then one of:
   - **Mamba-2** (40 blocks): the input projection gives the gate *z*, the conv input *x B C* and the step sizes; a causal convolution of width 4; then per head `S = S·exp(dt·A) + dt·x·Bᵀ` and `y = S·C + D·x`; a gated RMSNorm per group of 1024; the output projection.
   - **Attention** (8 blocks): q/k/v projections, K/V appended to the cache, grouped-query softmax attention over every earlier token, o projection.
   - **MoE** (40 blocks): the router picks 22 of 512 experts (sigmoid scores, a bias that only shifts the choice); the latent projection compresses the input to 1024; resident experts run first, missing ones as they arrive from the SSD; their weighted sum goes back to 4096; the shared expert adds its output.
   
   The block's output is added to the residual.
3. **Head.** The final RMSNorm and the LM head give the logits; a token is sampled.
4. **Draft.** The MTP blocks draft the next token(s); the next step verifies them in a single pass.

## One prefill chunk (up to 512 tokens)

Same block structure, batched. FP8 projections decode tiles of rows to F16 once and multiply them by all the tokens; NVFP4 matrices use the int8 VNNI GEMM. Tokens are grouped by expert, so every needed expert is read once per chunk and applied to all of its tokens. The Mamba recurrence advances token by token inside the chunk, parallel over its 128 heads.

## Sequence state

The attention caches are positional. The Mamba state and its convolution history are not, they summarize the whole prefix. Three mechanisms keep this cheap:
- **checkpoint:** before the last prompt token of each request, so a resent conversation that diverges after that point does not recompute the prompt;
- **speculative snapshots:** after each draft position of a verify pass, so a rejected draft costs a copy;
- **MTP cache:** filled one position behind the main model.


## Threading

`tp_for` hands out chunks of rows through an atomic cursor, so fast P-cores simply take more chunks than E-cores and there are no stragglers.
- **Placement.** Workers are pinned, one per physical P or E core. SMT siblings and the low-power E-cores are left to the HTTP server and I/O completion.
- **Idle waiting.** Idle workers wait with `UMONITOR/UMWAIT` (WAITPKG) before falling back to a futex. On a laptop this keeps the turbo budget for the cores doing work.

## Correctness strategy

- **Reference.** `tests/ref_nemotron.py` runs the official transformers implementation (`nemotron_h`) on the official NVFP4 checkpoint, one block at a time in float32. The routed experts are replaced by a lazy loader with the same math, so only the experts the router picks are loaded. The MTP head, which transformers does not implement, follows vLLM.
- **Block by block.** `INZ_DUMP_HIDDEN=file inz dump` writes the engine's residual after every block. `tests/compare_hidden.py` compares it with the reference's `--dump-hidden`. A bug shows up as a jump at the block that has it.
- **Logits.** `tests/compare_logits.py` gives KL, cosine, top-1 and top-5 agreement.
- **Byte-exact parity tests:**
  - tokenizer vs HF `tokenizers` (`tests/tokenizer_parity.py`);
  - chat template vs Jinja (`tests/chat_parity.py`).
- **Code quality.** `tests/code_eval.py` gives the served model C and JavaScript tasks with hidden tests. Its `--selftest` checks the tests against reference solutions.
