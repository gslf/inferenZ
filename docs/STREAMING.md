# SSD expert streaming

[README](../README.md)

The routed experts of Laguna S 2.1 are 113B of its 118B parameters: 256 experts in each of 47 MoE layers, top-10 per token. The converter writes each expert as one contiguous **blob**, the gate|up matrix (2048 × 3072) followed by the down matrix (3072 × 1024). There are two kinds of blobs:

| Layers | Format | Blob | On SSD |
|---|---|---:|---:|
| 1-39 | Q4N (the checkpoint's NVFP4) | 5.07 MiB | 52.8 GB |
| 40-47 | Q8G (the checkpoint's BF16) | 9.28 MiB | 19.9 GB |

- Each blob is padded to 4 KiB, and the GGUF alignment is 4 KiB, so every blob can be read with a single `O_DIRECT` request straight into its cache slot.
- Large reads are where the drive delivers its bandwidth.
- *LLM in a flash* (Apple, 2024) calls this row-column bundling. ds4, flash-moe and SSD-LLaMA all converge on the same layout.

The model directory should be `chattr +C` on btrfs (no copy-on-write, no compression), otherwise O_DIRECT falls back to buffered I/O.

## The cache

- **Arena.** One anonymous, huge-page-advised arena divided into fixed slots of one blob each. The budget is automatic: the engine's whole footprint stays within `MemTotal − reserve` (and within what is free), or `--cache 40G`. The reserve (`--ram-reserve`, default max(10 GB, 30% of RAM)) keeps the desktop and the page cache out of swap.
- **One cache per blob size.** Each blob kind has its own arena of fixed slots and its own eviction; the memory budget is split in proportion to their sizes, so both keep the same fraction of their experts.
- **I/O.** Reads go through `io_uring` with a registered file. Fixed buffers are used when `RLIMIT_MEMLOCK` allows; optional SQPOLL can be pinned to a low-power core.
- **Two queues.** *Demand* reads (a layer is waiting) are always submitted before *prefetch* reads.
- **Eviction (LRFU).** Each expert keeps a decayed-frequency score `crf = crf·2^(−λΔt) + w`, where `w` is an access weight and `Δt` counts tokens processed by the main model. λ moves it between LFU and LRU. Scores survive eviction, so frequency is remembered. Default `λ = 0.01` (half-life 100 tokens).
- **Victim choice.** As in Redis's approximated LRU, a victim is chosen by sampling 24 random slots and evicting the lowest score, without maintaining an ordered structure. Prefetched-but-unused blobs are discounted.
- **Pinning.** Slots in use by the current layer are pinned and never evicted.
- **Warm start.** The scores are saved at exit (`~/.cache/inferenz/<model>.<size>.scores`) and the highest-scoring experts are preloaded at the next start (`--hotlist`).
- **Accounting.** Hits and misses are counted once, at routing time.

## Hiding the SSD behind compute

For each MoE layer of a decode step:

1. **Route** (sigmoid over 256, top-10 after a selection bias, renormalized, scaled by 2.5).
2. **Issue demand reads** for the non-resident experts of this layer.
3. **Compute the shared expert** while the reads are in flight.
4. **Compute available experts together**, including new arrivals, without waiting for every read.
5. **Combine the results in a fixed order**, so arrival order does not change the answer.
6. **Router lookahead** (`--lookahead 1`, default). The updated residual goes through the next layer's MLP norm and router, and its top-10 are queued as prefetches while that layer's attention runs. It works because the residual stream changes slowly from layer to layer (Eliseev & Mazur 2023, SpecMD). The cost is one 256 × 3072 matvec per layer.
Per-layer timing, stall seconds, read latency histograms, lookahead precision and recall are all in the diagnostics.

On the reference laptop (62 GB, default reserve) the caches hold about half of the experts; [COMPARISON.md](COMPARISON.md) has the measured hit rates.

## Prefill

Prefill processes the prompt in chunks of up to 512 tokens. For each layer:
- tokens are grouped by expert (expert-major);
- each expert is read once per chunk and applied to all of its tokens.

Short prompts are the hard case. With top-10 of 256, 100 tokens already touch 256 · (1 − (246/256)^100) ≈ 251 distinct experts per layer, each for a token or a few. A short prompt is therefore bound by the SSD unless most experts are resident.

## Routing traces

`--route-trace FILE` records every routing decision (position, layer, experts, weights). `tools/cachesim` replays a trace token by token against several policies, Belady's OPT included, to choose the eviction policy and to predict the hit rate of a cache size.
