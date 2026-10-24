# Quantization

[README](../README.md) · [Bottlenecks](BOTTLENECKS.md)


To produce each token, the model reads gigabytes of weights from RAM, and on a laptop the speed of memory is the limit. Storing each weight in fewer bits means fewer bytes to read per token, faster generation, a smaller file, and more experts that fit in RAM.

The price is precision. A weight stored in 4 bits is only an approximation of the original. The error has to stay small enough not to change what the model writes. inferenZ measures it for every recipe, by comparing its predictions with those of the official model.

## The formats

Weights are stored in groups of 64. In the quantized formats, each group shares one scale factor, and each weight in the group is stored with a few bits.

| Format | Bits per weight | What it is | Used for |
|---|---:|---|---|
| F32 | 32 | exact | small tensors that control gates and decay rates |
| F16 | 16 | exact: holds the original weights without loss | the exact recipe |
| FP8 | 8 | the original 8-bit format, kept as is | the n-gram table |
| Q8G | 8.25 | 8-bit integers | dense weights in the recommended recipe, experts in the exact one |
| Q4L | 4.25 | 4-bit index into a table of 16 levels | experts in the recommended recipe |
| Q3L, Q2L | 3.25, 2.25 | 3- and 2-bit versions of Q4L | available, not used by either recipe |

## Three ideas that keep the error small

**Rotation.** Some inputs of a layer carry a few very large values. In a group that contains one, the shared scale has to be large, and every other weight in the group loses precision. Before quantizing, the weights are rotated with a fixed pseudo-random transform that spreads those large values evenly over the group. At run time the input of the layer is rotated the same way, so the result does not change.

**Learned levels.** A 4-bit weight can take 16 values. Instead of spacing them evenly, the converter chooses them for each tensor, to fit how its weights are actually distributed: more levels where weights are frequent, fewer at the extremes.

**Importance.** Not all inputs matter equally. The engine can run the model on a sample text and record how strongly each input is used; the converter then spends more precision where it matters (an *importance matrix*).

## Why decoding is cheap

The table of levels is so small that the CPU looks up 32 weights with a single instruction, and multiplies them by the layer's input with another (the AVX-VNNI instructions of recent x86 CPUs). Turning a 4-bit weight back into a number therefore costs almost nothing: the speed is set by how fast memory delivers the bytes. On the reference laptop, the 4-bit kernel reads weights at 74 GB/s, close to the RAM's peak of 77 GB/s.

## The LM head in two stages

To pick each token, the model scores all 248,320 tokens of its vocabulary, which means reading 1.27 GB of weights at full precision. Most of those tokens are obviously wrong, so computing them exactly is wasted work. inferenZ works in two stages:
1. a 4-bit copy of the head (0.34 GB) gives every token a rough score;
2. the tokens whose rough score is close to the best one, about 240 of them, are recomputed exactly from the full-precision weights;
3. the other tokens keep their rough score, slightly lowered so that, together, they do not get more probability than they should.

The tokens that can actually be chosen are always scored exactly. Measured on the full model, the two stages change the predictions by a KL of 0.003, much less than the rest of the recipe.

## Prompts use a different kernel

Generating a token is limited by memory, because each weight is used once. Reading a prompt is limited by computation, because each weight is used for every token of the prompt. For 12 tokens or more, inferenZ converts blocks of weights to plain 8-bit integers once and multiplies them by several tokens at a time. On the reference laptop this reaches 1.2 to 2.3 trillion operations per second, up to about half of what the CPU can do.

## Choosing the recipe

The measurements on the full model (see the [README](../README.md#get-the-model-and-convert-it)) led to one rule:
- **4-bit experts** cost little;
- **4-bit dense weights cost much more**, above all the attention projections, which are the most sensitive weights of the model.

The recommended recipe therefore keeps every dense weight at 8 bits and stores the experts at 4 bits: a 124 GB file, KL 0.014 against the official model, and the same top prediction 96.8% of the time.
