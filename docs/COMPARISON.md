# Nemotron 3.5 Super vs Qwen3.8-Flash-Next

[README](../README.md)

This branch (`nemotron-3.5`) runs NVIDIA Nemotron 3.5 Super; `main` runs Qwen3.8-Flash-Next. Both were measured on the reference laptop (Core Ultra 7 265U, 62 GB, NVMe), on 2026-11-04, with the same tools and settings.

## The two models

| | Qwen3.8-Flash-Next (`main`) | Nemotron 3.5 Super (this branch) |
|---|---|---|
| parameters | 125 B, 6 B active per token | 120 B, 12 B active per token |
| context mixing | 36 Gated DeltaNet + 12 sparse attention | 40 Mamba-2 + 8 attention |
| experts | 512 per layer, top-10 | 512 per block, top-22 (LatentMoE) |
| weights in inferenZ | q4q8 recipe: experts 4-bit LUT, dense 8-bit (lossy) | the official NVFP4 release, copied exactly |
| file | 124 GB | 73 GB |
| bytes read per token | ~6 GB | ~10 GB |
| accuracy vs the official model | KL 0.014, top-1 96.8% (q4q8) | KL 0.0031, top-1 96.1% |

The accuracy rows compare each engine with its own model's official implementation in float32 (Qwen: the previous session's measurement on its own corpus; Nemotron: 51 tokens of a chat prompt, plus KL 0.0026 for the MTP head).

## Speed

`tools/bench_compare.py`, server defaults, warm expert cache (scores of an earlier session), greedy, no thinking, 5 prompts × 256 tokens, then a 3.8k-token prompt.

| | Qwen3.8 | Nemotron 3.5 |
|---|---:|---:|
| decode, plain | 7.53 tok/s | 3.95 tok/s |
| decode, MTP (`--spec-k 1`) | 9.74 tok/s | 4.71 tok/s |
| MTP drafts accepted | 86% | 91% |
| prefill, 3.8k-token prompt | 34.9 tok/s | 16.5 tok/s |
| time to first token, 3.8k-token prompt | 110 s | 232 s |
| prefill, short prompts (~60 tokens) | 21.0 tok/s | 7.2 tok/s |
| expert cache hit rate | 92.8% | 84.8% |
| expert cache | 35.5 GiB, 58% of the experts | 32.8 GiB, 54% of the experts |
| dense weights in RAM | 4.3 GB | 7.2 GB |
| process RSS | 43.6 GB | 43.9 GB |

Nemotron reads twice the bytes per token (12 B active parameters, FP8 Mamba projections, 22 experts per block of 3 MB each), so it decodes at about half the speed. Its MTP gains less (+19% vs +29%) although more drafts are accepted: a two-token verify touches up to 44 experts per block, so it costs 1.5× a decode step.

## Coding quality

`tests/code_eval.py`: 20 C and JavaScript tasks with hidden tests (C built with `-Wall -Wextra -Werror` and ASan/UBSan), `reasoning_effort: medium` (Qwen: its medium effort; Nemotron: its *efficient* thinking), temperature 1.0, 16k tokens, two samples per task.

| | Qwen3.8 | Nemotron 3.5 |
|---|---:|---:|
| sample 1 | 17/20 | 15/20 |
| sample 2 | 16/20 | 12/20 |
| total | 33/40 (82.5%) | 27/40 (67.5%) |
| C tasks | 15/20 | 14/20 |
| JavaScript tasks | 18/20 | 13/20 |
| average answer (reasoning + code) | 2,155 tokens, 3.8 min | 1,903 tokens, 7.0 min |

Both models fail `c_parse_long` in every sample for the same reason: they use `NULL` without including a header that defines it. Nemotron's other failures are mostly hidden JavaScript edge cases (`js_flatten`, `js_template` in both samples) and C code that does not compile under `-Werror` (`c_hashmap`). With two samples per task the totals carry a few points of noise, but the gap (6 tasks out of 40) is consistent with the published benchmarks below.

## Published benchmarks

NVIDIA's model card compares the two models (BF16 Nemotron, the vendors' own harnesses). Qwen3.8-Flash-Next is ahead on most rows, e.g. GPQA-Diamond 92.7 vs 89.5, HLE 39.6 vs 31.8, LiveCodeBench v6 94.1 vs 88.2, SWE-bench Verified 78.7 vs 71.1. Nemotron's NVFP4 release scores within a point of its BF16 weights on nearly every row.

## Verdict

On this laptop Nemotron 3.5 Super is about half as fast as Qwen3.8-Flash-Next (4.7 vs 9.7 tok/s with MTP, twice the time to the first token) and solved fewer coding tasks (67.5% vs 82.5%), in line with the published benchmarks. Its advantages are a smaller file (73 vs 124 GB), weights that are exactly the vendor's with no recipe to tune, and an engine that matches the official implementation more closely (KL 0.0031 vs 0.014). For this machine and these goals, Qwen3.8-Flash-Next on `main` remains the better model.
