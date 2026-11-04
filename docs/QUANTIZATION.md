# Weight formats

[README](../README.md)

To produce each token, the model reads gigabytes of weights from RAM, and on a laptop the speed of memory is the limit. Storing each weight in fewer bits means fewer bytes to read per token, faster generation, a smaller file, and more experts that fit in RAM. The price is precision.

Nemotron 3.5 Super is published already quantized: NVIDIA trained the NVFP4 release by distillation from the BF16 weights, and its published benchmarks match them. inferenZ therefore keeps every published number exactly. There is no recipe to choose and nothing to calibrate: the engine runs the model NVIDIA released.

## The formats

| Format | Bits per weight | What it is | Used for |
|---|---:|---|---|
| Q4N | 4.5 | NVFP4 repacked without changing a value: 4-bit codes (e2m1), one 8-bit scale per 16 weights, one scale per matrix | routed and shared experts |
| F8T | 8 | FP8 (e4m3) copied as is, one scale per matrix | Mamba and attention projections, LM head |
| F16 | 16 | the BF16 originals widened (F16 has more mantissa bits; only magnitudes below 6·10⁻⁵ lose relative precision, by less than 3·10⁻⁸) | router, latent projections, MTP projections, embeddings |
| F32 | 32 | exact | norms, convolutions, step sizes |

The LUT formats of the engine (Q8G, Q4L, Q3L, Q2L, with rotation, learned levels and importance weighting) remain available to `inz convert` for studies, but the default conversion does not use them.

## How the kernels read the exact formats

**Q4N.** A 4-bit code selects one of 16 values with a single PSHUFB, and AVX-VNNI multiplies 32 of them by the int8 input in one instruction. Every 4 weights of a 16-weight block share a 32-bit lane, so each lane is scaled by its block's e4m3 scale. The scale is decoded by moving its bits into an fp16 number: exact for zero and subnormal scales too, which the checkpoint contains.

**F8T.** Sixteen e4m3 bytes become sixteen fp16 bit patterns with three integer instructions (`(b << 8) >> 1`, arithmetic, then a mask): the same numbers divided by 2⁸, subnormals included. The CPU converts them to f32 and multiplies them by the f32 input, so the result is exact up to f32 rounding. The input is not quantized. One P-core reads 19 GB/s of FP8 weights, so ten cores outrun the RAM.

**Bit-identical decode and verify.** A row of an F8T matrix is summed in the same order whether one token or several are computed, so speculative verification gives the same logits as plain decoding.

## Prompts use other kernels

Generating a token is limited by memory, because each weight is used once. Reading a prompt is limited by computation, because each weight is used for every token of the prompt. For 12 tokens or more:
- Q4N matrices are converted, a tile at a time, to 8-bit integers and multiplied by several tokens with VNNI;
- F8T matrices are converted, a few rows at a time, to F16 and multiplied by blocks of tokens with FMA.

## Precision of the activations

The activations stay f32 between operators. Q4N matrices quantize their input to int8 with one scale per 64 values; F8T, F16 and F32 matrices use the f32 input directly. NVIDIA's own serving on Blackwell GPUs quantizes the activations to 4 bits (NVFP4) or 8 bits (FP8), so inferenZ computes with more precision than the deployment the model was tuned for.
