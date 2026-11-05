# inferenZ

**A single-model inference engine in C for Poolside Laguna S 2.1 (118B parameters, 8.5B active), running on CPU with experts streamed from the SSD.**

Linux only. CPU only for now.

Laguna S 2.1 is a mixture-of-experts coding model:
- 48 layers: 12 with global attention (YaRN RoPE, up to 1M tokens), 36 with a 512-token sliding window;
- 256 experts per layer (top-10) plus a shared expert, from layer 1 on.

inferenZ runs the official NVFP4 release: its 4-bit experts exactly as published, the rest at 8 bits.


## Quick start

Requirements: Linux on x86-64 with AVX2 and AVX-VNNI, about 64 GB of RAM, gcc ≥ 15, liburing and OpenMP. The model must already be converted: see [Get the model and convert it](#get-the-model-and-convert-it) (the default file is `models/laguna-s-2.1.inz.gguf`).

```sh
make
./inz serve
```

Open <http://127.0.0.1:8080> once the server prints `serving on http://127.0.0.1:8080`.

- **Start-up** takes about a minute: the dense weights load, then the expert cache is warmed with the experts used in earlier sessions (`~/.cache/inferenz/`).
- **The first answers are slower** while the cache adapts to what you ask.
- **Stop** with `Ctrl+C`. The server saves which experts were used, so the next start is faster.

### The web UI

- **Thinking**: *on* (the model's default) thinks before answering; *off* answers directly.
- **Monitor** shows live speed, expert-cache hit rate and where each token's time goes.

### The terminal

```sh
./inz run --effort none
```

Type a message and press Enter; `/quit` exits.

### Other programs

The same server is an OpenAI-compatible API at `http://127.0.0.1:8080/v1` (any API key works unless `--api-key` is set). It supports tool calls, so an agent harness can give the model its own tools (running code, reading files, etc.).

```sh
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Ciao!"}],"reasoning_effort":"none"}'
```

### Speed on the reference laptop

See [the comparison](docs/COMPARISON.md) with Qwen3.8-Flash-Next (`main`) and Nemotron 3.5 Super (`nemotron-3.5`).

### Memory

The engine sizes its expert cache automatically and leaves max(10 GB, 30% of RAM) to the rest of the system. If the minimum cache does not fit, startup stops with a clear error.

- **`Killed` at start-up, or a sluggish desktop**, means the system ran short of memory: close other large programs, or give the engine less with `--cache 30G` or a larger `--ram-reserve`.
- **On a machine doing nothing else**, `--ram-reserve 8` leaves more room to the cache, which is faster.



## References

- **Laguna S 2.1 vs Qwen3.8-Flash-Next and Nemotron 3.5 Super** (the `main` and `nemotron-3.5` branches): speed, accuracy and coding quality on the same laptop. [docs/COMPARISON.md](docs/COMPARISON.md)
- **Architecture overview.** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
- **Diagnostics.** [docs/DIAGNOSTICS.md](docs/DIAGNOSTICS.md)
- **OpenAI-compatible API**: chat (reasoning, tools, streaming), completions. [docs/API.md](docs/API.md)
- **The model.** [How it works](docs/MODEL.md).
- **Precision and speed.** [Weight formats](docs/QUANTIZATION.md): NVFP4 experts kept exactly, the rest at 8 bits.
- **SSD expert streaming.** One 4 KiB-aligned blob per expert, `io_uring` + `O_DIRECT`, an LRFU cache, router lookahead prefetch, and missing experts computed in arrival order. [docs/STREAMING.md](docs/STREAMING.md)
- **No speculative decoding**: Laguna has no MTP head and its drafter is a separate model. [docs/SPECULATIVE.md](docs/SPECULATIVE.md)



## Get the model and convert it

`tools/hfget.py` downloads the official NVFP4 checkpoint (100 GB) in priority order: dense weights first, then the experts layer by layer. It resumes after interruptions. The converter can start while the download is still running.

```sh
mkdir -p models && chattr +C models          # btrfs: no CoW/compression, so O_DIRECT works
python3 -I tools/hfget.py poolside/Laguna-S-2.1-NVFP4 models/Laguna-S-2.1-NVFP4 --ipv4
./inz convert --src models/Laguna-S-2.1-NVFP4 --out models/laguna-s-2.1.inz.gguf
```

The conversion repacks the NVFP4 experts exactly and stores the rest at 8 bits (router and embeddings F16) in a 77 GB file. `--attn F16 --dense F16 --shared F16 --head F16` keeps the BF16 tensors exact, at about twice their bytes per token.



## License

The code is under the MIT license ([LICENSE](LICENSE)).

- `src/unicode_tables.inc` is generated from the Unicode Character Database (© Unicode, Inc., Unicode License v3).
- The model weights are not part of this repository and are under their own license (OpenMDW-1.1, see the model card).
