# Laguna S 2.1 vs Qwen3.8-Flash-Next vs Nemotron 3.5 Super

[README](../README.md)

This branch (`laguna-s-2.1`) runs Poolside Laguna S 2.1; `main` runs Qwen3.8-Flash-Next and `nemotron-3.5` runs NVIDIA Nemotron 3.5 Super. All three were measured on the reference laptop (Core Ultra 7 265U, 62 GB, NVMe), on 2026-11-04 (Qwen, Nemotron) and 2026-11-05/06 (Laguna), with the same tools and settings.

## The three models

| | Qwen3.8-Flash-Next (`main`) | Nemotron 3.5 Super (`nemotron-3.5`) | Laguna S 2.1 (this branch) |
|---|---|---|---|
| parameters | 125 B, 6 B active per token | 120 B, 12 B active per token | 118 B, 8.5 B active per token |
| context mixing | 36 Gated DeltaNet + 12 sparse attention | 40 Mamba-2 + 8 attention | 36 sliding-window (512) + 12 global attention |
| experts | 512 per layer, top-10 | 512 per block, top-22 (LatentMoE) | 256 per layer, top-10, plus a shared expert |
| weights in inferenZ | q4q8 recipe: experts 4-bit LUT, dense 8-bit (lossy) | the official NVFP4 release, copied exactly | the official NVFP4 experts copied exactly; BF16 experts (layers 40-47) and dense tensors at 8 bits |
| file | 124 GB | 73 GB | 77 GB |
| bytes read per token | ~6 GB | ~10 GB | ~6.7 GB |
| accuracy vs the official model | KL 0.014, top-1 96.8% | KL 0.0031, top-1 96.1% | KL 0.019, top-1 95.7% |
| speculative decoding | built-in MTP head | built-in MTP head | none (Poolside's drafter is a separate model) |

The accuracy rows compare each engine with its own model's official implementation in float32: Qwen on the previous session's corpus, Nemotron on 51 tokens of a chat prompt, Laguna on 512 tokens of a document. The models are not equally sensitive to rounding. Laguna amplifies it most: its own float32 reference, with only the experts' inputs rounded to int8 as the engine rounds them, already scores KL 0.017. The W4A4 serving its checkpoint was made for scores KL 0.045 ([QUANTIZATION.md](QUANTIZATION.md#accuracy-of-the-whole-model)).

## Speed

`tools/bench_compare.py`, server defaults, warm expert cache (scores of an earlier session), greedy, no thinking, 5 prompts × 256 tokens, then a long prompt (the same 14,000 characters of documentation for every model).

| | Qwen3.8 | Nemotron 3.5 | Laguna S 2.1 |
|---|---:|---:|---:|
| decode, plain | 7.53 tok/s | 3.95 tok/s | 6.01 tok/s |
| decode, MTP (`--spec-k 1`) | 9.74 tok/s | 4.71 tok/s | (no MTP) |
| MTP drafts accepted | 86% | 91% | |
| prefill, long prompt | 34.9 tok/s (3.8k tokens) | 16.5 tok/s | 27.3 tok/s (4.3k tokens) |
| time to first token, long prompt | 110 s | 232 s | 156 s |
| prefill, short prompts (~60-110 tokens) | 21.0 tok/s | 7.2 tok/s | 9.6 tok/s |
| expert cache hit rate | 92.8% | 84.8% | 87.5% |
| expert cache | 35.5 GiB, 58% of the experts | 32.8 GiB, 54% of the experts | 35.1 GiB, 52% of the experts |
| dense weights in RAM | 4.3 GB | 7.2 GB | 3.9 GB |
| process RSS | 43.6 GB | 43.9 GB | 43.2 GB |

Laguna's tokenizer splits the long prompt into 4,286 tokens (Qwen: 3.8k), so its time to first token is the fairer comparison.

Laguna's decode is limited by RAM, not by the SSD. Every token reads 3.9 GB of attention, shared-expert and head weights plus 2.9 GB of experts. A cold start (no scores from an earlier session) decoded as fast as a warm one (5.47 vs 5.48 tok/s, before the change below): the cache reaches an 88% hit rate within the warm-up prompts. A 32-token trace during a coding answer gives 150 ms per token: 62 ms of attention, 51 ms of experts already in RAM, 10 ms of shared experts and 13 ms for the experts read from the SSD.

This branch made that decode 10% faster (5.48 → 6.01 tok/s) without changing a single result ([QUANTIZATION.md](QUANTIZATION.md#how-the-kernels-read-them), [ARCHITECTURE.md](ARCHITECTURE.md#threading)). The kernels now prefetch the next rows, so short rows stream as fast as long ones, and one token's input is quantized in parallel inside the dispatch that uses it. The other branches use the same kernels without these changes, so their numbers above are before any such gain.

## Coding quality

`tests/code_eval.py`: 20 C and JavaScript tasks with hidden tests (C built with `-Wall -Wextra -Werror` and ASan/UBSan), `reasoning_effort: medium` (Qwen: its medium effort; Nemotron: its *efficient* thinking; Laguna: thinking on, its only reasoning mode), temperature 1.0, 16k tokens, two samples per task.

| | Qwen3.8 | Nemotron 3.5 | Laguna S 2.1 |
|---|---:|---:|---:|
| sample 1 | 17/20 | 15/20 | 12/20 |
| sample 2 | 16/20 | 12/20 | 14/20 |
| total | 33/40 (82.5%) | 27/40 (67.5%) | 26/40 (65%) |
| C tasks | 15/20 | 14/20 | 13/20 |
| JavaScript tasks | 18/20 | 13/20 | 13/20 |
| average answer (reasoning + code) | 2,155 tokens, 3.8 min | 1,903 tokens, 7.0 min | 5,401 tokens, 17.3 min |

Laguna thinks far longer than the other two: its median answer is 1,930 tokens, but 15 of its 40 answers exceed 7,000. Four of the 40 reached the 16k limit without writing any code (`c_csv_field` in both samples, `c_balanced` and `js_emitter` once each). Its reasoning stays coherent at that length (checked past the 4,096 positions where the sliding layers' caches wrap around): it keeps re-examining edge cases instead of concluding. Its quick failures are hidden edge cases: `deepEqual(0, -0)`, empty nested objects in `js_flatten` (both samples, as for Nemotron), and `c_base64` in both samples (a wrong decoder once, an initializer warning that `-Werror` rejects once). It is the only model that once solved `c_parse_long`, which Qwen and Nemotron failed in every sample.

## Published benchmarks

The vendors publish few common benchmarks. NVIDIA's model card compares Nemotron with Qwen3.8-Flash-Next (BF16 Nemotron, the vendors' own harnesses): Qwen is ahead on most rows, e.g. GPQA-Diamond 92.7 vs 89.5, HLE 39.6 vs 31.8, LiveCodeBench v6 94.1 vs 88.2, SWE-bench Verified 78.7 vs 71.1. Poolside's card reports agentic coding benchmarks only (SWE-bench Multilingual 78.5%, SWE-Bench Pro 59.4%, Terminal-Bench 2.1 70.2%), and does not list either of the other two models. Third-party aggregators put Qwen3.8-Flash-Next at 81% on SWE-bench Multilingual.

## Verdict

On this laptop Laguna S 2.1 sits between the other two on speed (6.0 tok/s, vs 7.5 for Qwen without MTP and 4.0 for Nemotron) and level with Nemotron on the coding tasks (65% vs 67.5%), well behind Qwen (82.5%). Its long reasoning makes it the slowest to answer by far: 17 minutes per task on average, against 4 for Qwen and 7 for Nemotron, and with no MTP head it cannot use speculative decoding. Its advantages are a 77 GB file (Qwen's recipe needs 124 GB), the vendor's NVFP4 experts used exactly, and agentic coding training (SWE-bench, Terminal-Bench) that these short single-turn tasks do not measure. For this machine and these goals, Qwen3.8-Flash-Next on `main` remains the better model.
