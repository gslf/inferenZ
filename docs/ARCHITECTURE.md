# Architecture

[README](../README.md)

inferenZ is a single-model engine. The model's shapes are compile-time constants (`src/model.h`), checked against the GGUF at load. Kernels, memory layout and scheduling are written for this model and for CPUs. The model itself is described in [MODEL.md](MODEL.md).

## Source map

| File | Role |
|---|---|
| `main.c` | CLI: `serve`, `run`, `bench`, `convert`, `info`, `dump` |
| `convert.c`, `safetensors.c` | official FP8 checkpoint → inferenZ GGUF; streams while the download is still running (reads the `hfget` journal) |
| `quant.c` | LUT quants, Hadamard rotation, quantizer; matvec, batched, int8 GEMM and F16 GEMM kernels ([QUANTIZATION.md](QUANTIZATION.md)) |
| `model_fwd.c` | forward pass: hyper-connections, n-gram embedding, Gated DeltaNet, QSA attention, MoE, MTP layer, two-stage head |
| `expert_cache.c` | io_uring + O_DIRECT expert cache with LRFU eviction ([STREAMING.md](STREAMING.md)) |
| `engine.c` | loading, memory budget, sequence state and checkpoints, generation and speculative loops ([SPECULATIVE.md](SPECULATIVE.md)) |
| `threadpool.c` | hybrid-CPU thread pool |
| `diag.c` | stage / phase / roofline counters, Perfetto traces, Prometheus ([DIAGNOSTICS.md](DIAGNOSTICS.md)) |
| `tokenizer.c` | NFC + Qwen pre-tokenizer split + byte-level BPE |
| `chat.c` | C port of the chat template; reasoning / content / tool-call stream parser |
| `server.c`, `http.c` | OpenAI-compatible API, SSE, web UI ([API.md](API.md)) |
| `gguf.c`, `json.c`, `sampler.c`, `util.c` | infrastructure |

## Memory tiers

What lives where, from the fastest tier to the slowest:

| Tier | Data | How it is held |
|---|---|---|
| RAM | dense weights, read for every token | one anonymous arena, loaded once, never paged out |
| RAM | expert cache: the experts in use | huge-page arena with LRFU eviction, as many experts as the memory budget allows |
| RAM | recurrent state | the DeltaNet state (113 MB, f32), its checkpoint and the speculative snapshots |
| RAM | attention caches | K/V in f16 and the indexer keys, for 13 QSA layers (12 + the MTP layer); pages are touched only as the context grows |
| SSD | the other experts | one 4 KiB-aligned blob per expert, one `O_DIRECT` read per miss |
| SSD | n-gram table (48 GiB) | memory-mapped; 16 random rows of 160 bytes per token, the hot ones stay in the page cache |
| SSD | embeddings and exact LM-head rows | read with `pread` per token, through the page cache |

## One decode step

1. **Embedding.** Read the token's row and copy it into the 4 residual streams.
2. **For each of the 48 layers:**
   1. *Layer 1 only:* the n-gram embedding. 16 hashed rows go through the key and value projections, are gated by the residual, and pass through a dilated convolution.
   2. *Hyper-connection in:* group RMSNorm of the 4 streams, then a low-rank mix gives the block's input (one stream wide) and the injection weights.
   3. *Mixer block*, one of two kinds:
      - **Gated DeltaNet** (36 layers): q/k/v and z projections, a short convolution, then a delta-rule update of each head's 128 × 128 state;
      - **QSA attention** (12 layers): q/k/v and indexer projections, the top-512 blocks of 4 tokens plus the unfinished block, then gated grouped-query attention.
   4. *Hyper-connection out:* each residual stream adds the block output times its injection weight.
   5. *MoE block*, inside its own hyper-connection: the router picks 10 of 512 experts (softmax); resident experts run first, missing ones as they arrive from the SSD; the shared expert adds its output through a sigmoid gate.
3. **Head.** The final mixer folds the 4 streams into one; the LM head (exact, or screen plus exact candidates) gives the logits; a token is sampled.
4. **Draft.** The MTP layer drafts the next token(s); the next step verifies them in a single pass.

## One prefill chunk (up to 512 tokens)

Same layer structure, batched. Projections use the F16 GEMM (exact recipe) or the int8 VNNI GEMM (quantized recipes). Tokens are grouped by expert, so every needed expert is read once per chunk and applied to all of its tokens. The DeltaNet recurrence advances token by token inside the chunk, parallel over its 48 heads. The n-gram rows of the whole chunk are gathered in parallel, so their page faults overlap on the SSD.

## Sequence state

The attention caches are positional. The DeltaNet state, its short convolutions and the n-gram convolution history are not, they summarize the whole prefix. Three mechanisms keep this cheap:
- **checkpoint:** before the last prompt token of each request, so a resent conversation that diverges after that point does not recompute the prompt;
- **speculative snapshots:** after each draft position of a verify pass, so a rejected draft costs a copy;
- **MTP cache:** filled one position behind the main model.


## Threading

`tp_for` hands out chunks of rows through an atomic cursor, so fast P-cores simply take more chunks than E-cores and there are no stragglers.
- **Placement.** Workers are pinned, one per physical P or E core. SMT siblings and the low-power E-cores are left to the HTTP server and I/O completion.
- **Idle waiting.** Idle workers wait with `UMONITOR/UMWAIT` (WAITPKG) before falling back to a futex. On a laptop this keeps the turbo budget for the cores doing work.

## Correctness strategy

- **Reference.** `tests/ref_qwen38.py` runs the official transformers implementation (`qwen4_exp`) on the official FP8 checkpoint, one layer at a time in float32. Two modules are replaced by lazy loaders with the same math: the n-gram table, so only the rows a prompt needs are read, and the routed experts, so only the experts the router picks are loaded.
- **Layer by layer.** `INZ_DUMP_HIDDEN=file inz dump` writes the engine's 4H residual after every layer. `tests/compare_hidden.py` compares it with the reference's `--dump-hidden`. A bug shows up as a jump at the layer that has it.
- **Logits.** `tests/compare_logits.py` gives KL, cosine, top-1 and top-5 agreement.
- **Byte-exact parity tests:**
  - tokenizer vs HF `tokenizers` (`tests/tokenizer_parity.py`);
  - chat template vs Jinja (`tests/chat_parity.py`).
- **Code quality.** `tests/code_eval.py` gives the served model C and JavaScript tasks with hidden tests. Its `--selftest` checks the tests against reference solutions.
