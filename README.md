# inferenZ

**A single-model inference engine in C for Qwen3.8-Flash-Next (125B parameters, 6B active), running on CPU with experts streamed from the SSD.**

Linux only. CPU only for now.

Qwen3.8-Flash-Next is a hybrid model:
- 48 layers: 36 Gated DeltaNet (linear attention with a recurrent state) and 12 sparse QSA attention layers;
- 4-stream hyper-connections instead of a single residual;
- a 512-expert MoE in every layer (top-10);
- a 51 GB hashed n-gram embedding;
- an MTP layer for speculative decoding.


## Quick start

Requirements: Linux on x86-64 with AVX2 and AVX-VNNI, about 64 GB of RAM, gcc ≥ 15, liburing and OpenMP. The model must already be converted: see [Get the model and convert it](#get-the-model-and-convert-it) (the default file is `models/qwen3.8-flash-next-q4q8.inz.gguf`).

```sh
make
./inz serve
```

Open <http://127.0.0.1:8080> once the server prints `serving on http://127.0.0.1:8080`.

- **Start-up** takes about a minute: the dense weights load, then the expert cache is warmed with the experts used in earlier sessions (`~/.cache/inferenz/`).
- **The first answers are slower** while the cache adapts to what you ask.
- **Stop** with `Ctrl+C`. The server saves which experts were used, so the next start is faster.

### The web UI

- **Thinking** sets the reasoning effort. At *high* the model may think for minutes before it answers; *medium* or *low* suit quick questions, *off* answers directly.
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

The engine sizes its expert cache automatically and leaves max(10 GB, 30% of RAM) to the rest of the system. If the minimum cache does not fit, startup stops with a clear error. On the 62 GB reference laptop that gives the cache about 35 GiB (57% of the experts) and the whole engine about 42 GB.

- **`Killed` at start-up, or a sluggish desktop**, means the system ran short of memory: close other large programs, or give the engine less with `--cache 30G` or a larger `--ram-reserve`.
- **On a machine doing nothing else**, `--ram-reserve 8` leaves more room to the cache, which is faster.



## References

- **Recent improvements.** [Speed and reliability](docs/ANALISI-2026-10-25.md).

- **Architecture overview.** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
- **Diagnostics.** [docs/DIAGNOSTICS.md](docs/DIAGNOSTICS.md)
- **OpenAI-compatible API**: chat (reasoning, tools, streaming), completions. [docs/API.md](docs/API.md)
- **The model.** [How it works](docs/MODEL.md). Quantized weights and the fast output screen are approximations; validation compares their predictions with the reference model.
- **Precision and speed.** [Weight formats and calibration](docs/QUANTIZATION.md).
- **SSD expert streaming.** One 4 KiB-aligned blob per expert, `io_uring` + `O_DIRECT`, an LRFU cache, router lookahead prefetch, and missing experts computed in arrival order. [docs/STREAMING.md](docs/STREAMING.md)
- **MTP speculative decoding** with exact speculative sampling. The 113 MB recurrent state is snapshotted during verification, so a rejected draft costs a copy. [docs/SPECULATIVE.md](docs/SPECULATIVE.md)



## Get the model and convert it

`tools/hfget.py` downloads the official FP8 checkpoint in priority order: dense weights first, then the n-gram table, then the experts layer by layer. It resumes after interruptions. The converter can start while the download is still running.

```sh
mkdir -p models && chattr +C models          # btrfs: no CoW/compression, so O_DIRECT works
python3 -I tools/hfget.py Qwen/Qwen3.8-Flash-Next-FP8 models/Qwen3.8-Flash-Next-FP8 --skip-vision

# 1. the exact recipe (also the reference for the others)
./inz convert --src models/Qwen3.8-Flash-Next-FP8 --out models/qwen3.8-flash-next-q8.inz.gguf --head F16 --embed F16

# 2. importance matrix: input statistics of every matrix on a calibration text (token ids, comma-separated)
INZ_IMATRIX=qwen38.imx ./inz dump -m models/qwen3.8-flash-next-q8.inz.gguf --exact-head --tokens @calib.txt --last 1 --out /dev/null

# 3. the recommended recipe
./inz convert --src models/Qwen3.8-Flash-Next-FP8 --out models/qwen3.8-flash-next-q4q8.inz.gguf --imatrix qwen38.imx --experts Q4L --dense Q8G --hc Q8G --shared Q8G --ple Q8G --router F16 --head F16 --screen Q4L --embed F16
```



## License

The code is under the MIT license ([LICENSE](LICENSE)).

- `src/unicode_tables.inc` is generated from the Unicode Character Database (© Unicode, Inc., Unicode License v3).
- The model weights are not part of this repository and are under their own license (see the model card).
