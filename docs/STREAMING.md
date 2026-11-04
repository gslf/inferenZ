# SSD expert streaming

[README](../README.md)

The routed experts of Nemotron 3.5 Super are 113B of its 120B parameters (plus 2.8B in the MTP block): 512 experts in each of 40 MoE blocks, top-22 per token. In NVFP4 (Q4N) that is 63.5 GB of 2.96 MiB blobs.

The converter writes each expert as one contiguous **blob**, the up matrix (2688 × 1024) followed by the down matrix (1024 × 2688), both in the latent space of the LatentMoE.

- Each blob is padded to 4 KiB, and the GGUF alignment is 4 KiB, so every blob can be read with a single `O_DIRECT` request straight into its cache slot.
- Large reads are where the drive delivers its bandwidth.
- *LLM in a flash* (Apple, 2024) calls this row-column bundling. ds4, flash-moe and SSD-LLaMA all converge on the same layout.

The model directory should be `chattr +C` on btrfs (no copy-on-write, no compression), otherwise O_DIRECT falls back to buffered I/O.

## The cache

- **Arena.** One anonymous, huge-page-advised arena divided into fixed slots of one blob each. The budget is automatic: the engine's whole footprint stays within `MemTotal − reserve` (and within what is free), or `--cache 40G`. The reserve (`--ram-reserve`, default max(10 GB, 30% of RAM)) keeps the desktop and the page cache out of swap.
- **One cache for all layers.** The 40 MoE blocks and the MTP MoE block (index 40) share the slots.
- **I/O.** Reads go through `io_uring` with a registered file. Fixed buffers are used when `RLIMIT_MEMLOCK` allows; optional SQPOLL can be pinned to a low-power core.
- **Two queues.** *Demand* reads (a layer is waiting) are always submitted before *prefetch* reads.
- **Eviction (LRFU).** Each expert keeps a decayed-frequency score `crf = crf·2^(−λΔt) + w`, where `w` is an access weight and `Δt` counts tokens processed by the main model. λ moves it between LFU and LRU. Scores survive eviction, so frequency is remembered. Default `λ = 0.01` (half-life 100 tokens).
- **Victim choice.** As in Redis's approximated LRU, a victim is chosen by sampling 24 random slots and evicting the lowest score, without maintaining an ordered structure. Prefetched-but-unused blobs are discounted.
- **Pinning.** Slots in use by the current layer are pinned and never evicted.
- **Warm start.** The scores are saved at exit (`~/.cache/inferenz/<model>.<size>.scores`) and the highest-scoring experts are preloaded at the next start (`--hotlist`).
- **Accounting.** Hits and misses are counted once, at routing time.

## Hiding the SSD behind compute

For each MoE block of a decode step:

1. **Route** (sigmoid over 512, top-22 after a selection bias, renormalized, scaled by 5).
2. **Issue demand reads** for the non-resident experts of this block.
3. **Compute the latent projection and the shared expert** while the reads are in flight.
4. **Compute available experts together**, including new arrivals, without waiting for every read.
5. **Combine the results in a fixed order**, so arrival order does not change the answer.
6. **Router lookahead** (`--lookahead 1`, default). The updated residual goes through the next MoE block's norm and router, and its top-22 are queued as prefetches while the Mamba or attention block in between runs. It works because the residual stream changes slowly from block to block (Eliseev & Mazur 2023, SpecMD). The cost is one 512 × 4096 matvec per block. Measured on the first nine blocks with half of their experts cached, it predicts 69% of the experts; `--lookahead 2` predicts from the MoE input instead, earlier but only 53% right, and was 6% slower.

Per-layer timing, stall seconds, read latency histograms, lookahead precision and recall are all in the diagnostics.

## Prefill

Prefill processes the prompt in chunks of up to 512 tokens. For each layer:
- tokens are grouped by expert (expert-major);
- each expert is read once per chunk and applied to all of its tokens.

Short prompts are the hard case. With top-22 of 512, 100 tokens already touch 512 · (1 − (490/512)^100) ≈ 506 distinct experts per block, each for a token or a few. A short prompt is therefore bound by the SSD unless most experts are resident.

## Routing traces

`--route-trace FILE` records every routing decision (position, layer, experts, weights). `tools/cachesim` replays a trace token by token against several policies, Belady's OPT included, to choose the eviction policy and to predict the hit rate of a cache size.
