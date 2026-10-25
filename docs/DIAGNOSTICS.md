# Diagnostics

[README](../README.md)

inferenZ keeps its diagnostics on in production. The server publishes a consistent snapshot between generated tokens and at the end of a request. During prefill, the monitor retains the last snapshot. They are exposed through:
- the web **Monitor** page;
- `GET /v1/diag` (JSON);
- `GET /metrics` (Prometheus);
- a **Perfetto trace** recorder;
- `inz bench` and `inz dump`.

## What is measured

### Per stage, per phase
The main stages record:
- TSC cycles;
- bytes streamed from RAM;
- bytes read from SSD;
- FLOPs.

Draft work and waiting for disk reads are measured separately, without counting their time twice. Expert read bytes are attributed to the layer that issues them, including reads anticipated for the next layer. Counters are kept separately for each **phase**: prefill, decode, speculative verify and draft. Decode and prefill have opposite bottlenecks, so mixing them hides both.

For each (phase, stage) the snapshot reports:

| Field | Meaning |
|---|---|
| `ms_per_token` | time per token of that phase |
| `gbs` | achieved bandwidth (bytes / time) |
| `roofline` | `gbs` ÷ **measured** peak DRAM bandwidth |
| `gflops` | achieved compute |
| `ssd_gb` | expert bytes that came from the SSD |

The peak is measured at startup with the engine's own pinned thread pool (768 MiB working set).
- A decode stage near 100% roofline is done: only fewer bytes can help.
- A stage at 60% is compute or latency bound, and its kernel is the work item.

### Per layer
Attention / experts / SSD-wait / other time for every layer. It also tracks the routing **adjacent-token overlap**, the fraction of a layer's experts reused by the next token, which bounds what any cache can do.

### Expert cache and SSD
| Metric | Description |
|---|---|
| hit rate | lifetime and recent hit rate, per layer |
| resident / slots | experts in RAM vs capacity |
| reads, bytes read | O_DIRECT reads issued through io_uring |
| read latency | average plus a log2 histogram |
| stall | seconds the compute threads waited on the SSD |
| prefetch | issued / useful / late / wasted (evicted unused) |
| evictions | count |

### Router lookahead
- **Precision**: fraction of prefetched experts that were actually selected.
- **Recall**: fraction of selected experts that had been prefetched.

These two numbers tell whether prefetch depth or width should change.

### Two-stage LM head
The running screening error σ and the average number of exact candidates per token.

### Speculative decoding
Drafted and accepted tokens and tokens per verify step (MTP).

### Threads and memory
Busy seconds per pinned thread (P/E imbalance shows here), RSS, available memory, and the sizes of dense weights, expert cache, recurrent state and attention caches. The position of the recurrent-state checkpoint is reported too.

## Tools

### Perfetto trace
```sh
curl -X POST 'http://127.0.0.1:8080/v1/diag/trace?tokens=16'
# ... generate ...
curl -o trace.json http://127.0.0.1:8080/v1/diag/trace
```
Open it in <https://ui.perfetto.dev>. Every stage of every layer of the next 16 tokens is a slice, so a missing-expert stall stands out as a red gap.

### Routing trace for offline cache studies
`--route-trace FILE` writes one line per (token, layer) with the selected experts. Feed it to any cache simulator (LRU / LFU / LRFU / Belady-OPT) to see how far the online policy is from optimal before touching the engine.