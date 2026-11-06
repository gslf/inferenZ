# Weight formats

[README](../README.md)

To produce each token, the model reads gigabytes of weights from RAM, and on a laptop the speed of memory is the limit. Storing each weight in fewer bits means fewer bytes to read per token, faster generation, a smaller file, and more experts that fit in RAM. The price is precision: inferenZ measures it by comparing its predictions with those of the official implementation running the official weights in float32.

## The recipe

| Tensors | Checkpoint | inferenZ | Why |
|---|---|---|---|
| experts, layers 1-39 | NVFP4 | Q4N: the same numbers, repacked | exact |
| experts, layers 40-47 | BF16 | Q8G | Poolside kept these layers at full precision in every quantized release; 8 bits keep them within about 1% (RMS) |
| attention, shared experts, layer 0 MLP, LM head | BF16 | Q8G | half the bytes of F16 for every token |
| router | BF16 | F16 | the expert choice is sensitive; 1.5 MB per layer |
| embeddings | BF16 | F16 | read one row per token |
| norms | BF16 | F32 | |

The 8-bit attention and MLP halve the bytes these tensors cost per token (3.8 GB instead of 7.3 GB) and cost no measurable accuracy over the whole model (below). `inz convert --attn F16 --dense F16 --shared F16 --head F16` keeps them exact.

The two expert formats live in two caches with different slot sizes ([STREAMING.md](STREAMING.md)).

## The formats

| Format | Bits per weight | What it is |
|---|---:|---|
| Q4N | 4.5 | NVFP4 repacked without changing a value: 4-bit codes (e2m1), one 8-bit scale per 16 weights, one scale per matrix. Gate and up share their global scale in the checkpoint, so they fuse into one matrix exactly |
| Q8G | 8.25 | 8-bit integers with one scale per 64 weights, after a fixed pseudo-random rotation that spreads outliers |
| F16, F32 | 16, 32 | exact for BF16 sources (F16 has more mantissa bits; only magnitudes below 6·10⁻⁵ lose relative precision, by less than 3·10⁻⁸) |

The LUT formats (Q4L, Q3L, Q2L) remain available to `inz convert` for studies.

## How the kernels read them

**Q4N.** A 4-bit code selects one of 16 values with a single PSHUFB, and AVX-VNNI multiplies 32 of them by the int8 input in one instruction. Every 4 weights of a 16-weight block share a 32-bit lane, scaled by its block's e4m3 scale (decoded exactly, zero and subnormal scales included).

**Q8G.** VNNI multiplies the int8 weights by the int8 input; one float scale per group.

**Prefetch.** The hardware prefetcher follows a stream only once it has seen a few of its lines. Q4N reads two rows at a time, so every row pair would start two new streams; Q8G starts one per row. Both kernels prefetch the next rows while they compute the current ones. Rows of 1024 to 3072 weights, the shapes of the experts, stream at 66-70 GB/s instead of 46-53 GB/s on the reference laptop.

**Bit-identical decode and verify.** Each row is summed in the same order whether one token or several are computed, so a rewound and recomputed position gives the same logits as plain decoding.

## Prompts use other kernels

Generating a token is limited by memory, because each weight is used once. Reading a prompt is limited by computation, because each weight is used for every token of the prompt. For 12 tokens or more, quantized matrices are converted, a tile at a time, to 8-bit integers and multiplied by several tokens with VNNI. A prompt is processed in chunks of 2048 tokens, so each expert sees enough tokens to amortize its tile decode.

## Precision of the activations

The activations stay f32 between operators. Quantized matrices quantize their input to int8 with one scale per 64 values (0.6% RMS rounding); F16 and F32 matrices use the f32 input directly. The checkpoint itself is made for W4A4 serving: its experts carry static input scales, and NVFP4 inference quantizes their inputs to 4 bits (about 10% RMS rounding).

## Accuracy of the whole model

Laguna amplifies small changes: a rounding in the experts' inputs grows from layer to layer, so no CPU format reaches the float32 reference exactly. `tests/ref_laguna.py --act int8|nvfp4` reruns the reference with the experts' inputs rounded as the engine rounds them, or as W4A4 serving does, which separates the cost of the activation format from any error of the engine.

512 tokens of a technical document, starting with the BOS token, compared with the official implementation in float32 on the same weights:

| | KL | top-1 agreement |
|---|---:|---:|
| inferenZ, the recipe above | 0.019 | 95.7% |
| inferenZ, dense tensors in F16 | 0.018 | 95.7% |
| the reference itself, experts' inputs rounded to int8 | 0.017 | 96.3% |
| the reference itself, experts' inputs rounded to NVFP4 (W4A4 serving) | 0.045 | 93.9% |

The engine is within 15% of the floor set by its int8 activations and closer to the reference than the W4A4 serving the checkpoint was made for. Storing the dense tensors exactly does not change the result, so the recipe keeps them at 8 bits.

Without the BOS token every prompt starts with, the model is far more sensitive: the same 512 tokens give KL 0.43 for the engine and 0.41 for the int8-rounded reference.
