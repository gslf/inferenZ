# Nemotron 3.5 Super

[README](../README.md)

NVIDIA Nemotron 3.5 Super has 120 billion parameters, but each token uses only about 12 billion of them. Most of the weights are *experts* that are consulted only when they are relevant. That is what lets a model this large run on a laptop: for each token, the engine only needs the small part of the model that the token uses.

inferenZ runs the language model of the official release, *Nemotron-3.5-Super-VL-120B-A12B*. The vision encoder of that release is not converted.

A token goes through 88 blocks. Every block reads the running state of the token (the *residual stream*), normalizes it, computes something and adds the result back. There are three kinds of blocks:
1. 40 **Mamba-2** blocks bring in context through a memory of fixed size;
2. 8 **attention** blocks bring in context by looking back at every earlier token;
3. 40 **MoE** blocks transform the token with a few experts chosen for it.

At the end the model gives a score to each of the 131,072 tokens of its vocabulary, and one of them is picked.

## Tokens in, tokens out

A tokenizer splits the text into tokens. Each token starts as a list of 4,096 numbers looked up in a table (the *embedding*). After the last block, the *LM head* turns the final numbers back into a score for every token of the vocabulary. The scores become probabilities, and the next token is drawn from them.

## Two ways of looking at the past

### Mamba-2: a compact memory (40 blocks)

A Mamba-2 block keeps a memory of fixed size instead of looking back at the tokens. The memory has 128 heads; each head stores a 64 × 128 table. For each new token:
- a short convolution mixes the token with the three before it;
- the memory fades a little, more or less depending on the token (the *step size*);
- the token writes into the memory what it carries, keyed by a 128-wide vector (*B*), and reads out what it needs with another one (*C*);
- the output is normalized and gated by the token itself, then projected back.

The cost per token is the same at token 10 as at token 100,000. The price is that the memory is a summary: it cannot quote an earlier passage word for word.

### Attention (8 blocks)

Eight blocks, spread through the model, use real attention, which can retrieve exact earlier tokens. 32 query heads share 2 key/value heads, and there is no positional encoding: the Mamba blocks already know where a token is. Attention reads every earlier token, so its cost grows with the length of the text, but only 8 blocks pay it.

Together, the Mamba blocks carry the gist of a long text cheaply, and the attention blocks fetch exact details when they are needed.

## Mixture of experts (LatentMoE)

Each MoE block has 512 small networks, the *experts*. A *router* scores all of them for the token (a sigmoid per expert); the 22 best are chosen, and their answers are weighted by their scores. The experts do not work on the 4,096 numbers of the token directly: a projection first compresses the token to 1,024 numbers (the *latent*), the experts work there, and a second projection brings their weighted sum back to 4,096. One more network, the *shared expert*, always runs on the full token.

The experts hold 113 of the 120 billion parameters, but a token uses only 22 of the 512 in each block. This is why inferenZ can keep the experts on the SSD and only the ones in use in RAM ([STREAMING.md](STREAMING.md)).

## MTP (speculative decoding)

The model comes with one extra attention block and one extra MoE block, trained to guess the token after the next one (*multi-token prediction*). inferenZ uses them to draft a token cheaply. The main model then checks the draft and produces its own next token in a single pass. When the draft is right, two tokens come out for little more than the cost of one. The output is exactly what the model would have written without the draft ([SPECULATIVE.md](SPECULATIVE.md)).

## In numbers

| | |
|---|---|
| parameters | 120 B, 113 B of them in experts; about 12 B used per token |
| blocks | 88: 40 Mamba-2, 40 MoE, 8 attention, plus the MTP blocks |
| numbers per token | 4,096 (1,024 inside the experts) |
| experts | 512 per MoE block, 22 chosen per token, plus 1 shared |
| vocabulary | 131,072 tokens |

## The official weights

NVIDIA publishes the model in two precisions. inferenZ uses the NVFP4 release (75 GB), which NVIDIA trained by distillation from the BF16 weights and whose published benchmarks match them:
- the experts and the shared experts in NVFP4: numbers of 4 bits, with a scale for every 16 of them;
- the Mamba and attention projections and the LM head in FP8: numbers of 8 bits, with one scale per matrix;
- everything else in BF16, numbers of 16 bits.

inferenZ converts them once into its own file, keeping every number exactly as published ([QUANTIZATION.md](QUANTIZATION.md)).
