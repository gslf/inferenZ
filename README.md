# inferenZ

**A single-model inference engine in C for NVIDIA Nemotron 3.5 Super (120B parameters, 12B active), running on CPU with experts streamed from the SSD.**

Linux only. CPU only for now.

Nemotron 3.5 Super is a hybrid model:
- 88 blocks: 40 Mamba-2 (a recurrent memory of fixed size), 8 attention, 40 mixture-of-experts;
- a LatentMoE: 512 experts per block, 22 chosen per token, working in a 1024-wide latent space;
- an MTP head for speculative decoding.

inferenZ runs the official NVFP4 release exactly as published: no weight is re-quantized.


## Quick start

Requirements: Linux on x86-64 with AVX2 and AVX-VNNI, about 64 GB of RAM, gcc ≥ 15, liburing and OpenMP. The model must already be converted: see [Get the model and convert it](#get-the-model-and-convert-it) (the default file is `models/nemotron-3.5-super.inz.gguf`).

```sh
make
./inz serve
```

Open <http://127.0.0.1:8080> once the server prints `serving on http://127.0.0.1:8080`.

- **Start-up** takes about a minute: the dense weights load, then the expert cache is warmed with the experts used in earlier sessions (`~/.cache/inferenz/`).
- **The first answers are slower** while the cache adapts to what you ask.
- **Stop** with `Ctrl+C`. The server saves which experts were used, so the next start is faster.

### The web UI

- **Thinking**: *on* (the model's default) thinks before answering; *efficient* asks the model for shorter reasoning; *off* answers directly.
- **Monitor** shows live speed, expert-cache hit rate, MTP acceptance and where each token's time goes.

### The terminal

```sh
./inz run --effort medium
```

Type a message and press Enter; `/quit` exits.

### Other programs

The same server is an OpenAI-compatible API at `http://127.0.0.1:8080/v1` (any API key works unless `--api-key` is set). It supports tool calls, so an agent harness can give the model its own tools (running code, reading files, etc.).

```sh
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Ciao!"}],"reasoning_effort":"medium"}'
```

### Memory

The engine sizes its expert cache automatically and leaves max(10 GB, 30% of RAM) to the rest of the system. If the minimum cache does not fit, startup stops with a clear error.

- **`Killed` at start-up, or a sluggish desktop**, means the system ran short of memory: close other large programs, or give the engine less with `--cache 30G` or a larger `--ram-reserve`.
- **On a machine doing nothing else**, `--ram-reserve 8` leaves more room to the cache, which is faster.



## References

- **Architecture overview.** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
- **Diagnostics.** [docs/DIAGNOSTICS.md](docs/DIAGNOSTICS.md)
- **OpenAI-compatible API**: chat (reasoning, tools, streaming), completions. [docs/API.md](docs/API.md)
- **The model.** [How it works](docs/MODEL.md).
- **Precision and speed.** [Weight formats](docs/QUANTIZATION.md): NVFP4 and FP8 kept exactly.
- **SSD expert streaming.** One 4 KiB-aligned blob per expert, `io_uring` + `O_DIRECT`, an LRFU cache, router lookahead prefetch, and missing experts computed in arrival order. [docs/STREAMING.md](docs/STREAMING.md)
- **MTP speculative decoding** with exact speculative sampling. The 168 MB recurrent state is snapshotted during verification, so a rejected draft costs a copy. [docs/SPECULATIVE.md](docs/SPECULATIVE.md)



## Get the model and convert it

`tools/hfget.py` downloads the official NVFP4 checkpoint (73 GB without the vision encoder) in priority order: dense weights first, then the experts block by block. It resumes after interruptions. The converter can start while the download is still running.

```sh
mkdir -p models && chattr +C models          # btrfs: no CoW/compression, so O_DIRECT works
python3 -I tools/hfget.py nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 models/Nemotron-3.5-Super-VL-NVFP4 --skip-vision
./inz convert --src models/Nemotron-3.5-Super-VL-NVFP4 --out models/nemotron-3.5-super.inz.gguf
```

The conversion copies every tensor exactly (NVFP4 → Q4N, FP8 → F8T, BF16 → F16) into a 74 GB file.



## License

The code is under the MIT license ([LICENSE](LICENSE)).

- `src/unicode_tables.inc` is generated from the Unicode Character Database (© Unicode, Inc., Unicode License v3).
- The model weights are not part of this repository and are under their own license (OpenMDW-1.1, see the model card).
