# Laguna S 2.1

[README](../README.md)

Poolside's Laguna S 2.1 has 118 billion parameters, but each token uses only about 8.5 billion of them. Most of the weights are *experts* that are consulted only when they are relevant. That is what lets a model this large run on a laptop: for each token, the engine only needs the small part of the model that the token uses. Laguna is trained for coding and agentic work.

A token goes through 48 layers. Each layer does two things:
1. it looks at the earlier tokens, to bring in context (*attention*);
2. it transforms what it has gathered, with a few experts chosen for this token (layer 0 uses one large network instead).

At the end the model gives a score to each of the 100,352 tokens of its vocabulary, and one of them is picked.

## Tokens in, tokens out

A tokenizer splits the text into tokens. Each token starts as a list of 3,072 numbers looked up in a table (the *embedding*). After the last layer, the *LM head* turns the final numbers back into a score for every token of the vocabulary. The scores become probabilities, and the next token is drawn from them.

## Two kinds of attention

Attention compares the current token with earlier ones and pulls in what is relevant. Its cost grows with the length of the text, so Laguna mixes two kinds:
- **global layers** (one in four, 12 in all) look at the whole context. They use 48 query heads, and their position encoding (RoPE, stretched with YaRN to a million tokens) covers half of each head;
- **sliding layers** (the other 36) look only at the last 512 tokens, with 72 query heads.

All layers share 8 key/value heads among their query heads (*grouped-query attention*). Every head's output is scaled by a learned gate that depends on the token, so a head can switch itself off when it has nothing useful to add.

The sliding layers keep the memory of a long conversation small: their caches hold a fixed window, and only the 12 global layers grow with the context.

## Mixture of experts

From layer 1 on, the transformation is done by 256 small networks, the *experts*. A *router* scores all of them for the token (a sigmoid per expert); the 10 best are chosen, and their answers are weighted by their scores. One more network, the *shared expert*, always runs.

The experts hold 113 of the 118 billion parameters, but a token uses only 10 of the 256 in each layer. This is why inferenZ can keep the experts on the SSD and only the ones in use in RAM ([STREAMING.md](STREAMING.md)).

## No built-in drafter

Laguna has no multi-token prediction head. Poolside's speculative decoding uses a separate draft model (DFlash), which inferenZ does not run, so Laguna generates one token per step.

## In numbers

| | |
|---|---|
| parameters | 118 B, 113 B of them in experts; about 8.5 B used per token |
| layers | 48: 12 global attention, 36 sliding-window attention (512 tokens) |
| numbers per token | 3,072 |
| experts | 256 per layer (layers 1-47), 10 chosen per token, plus 1 shared |
| vocabulary | 100,352 tokens |
| context | up to 1M tokens |

## The official weights

Poolside publishes BF16, FP8, INT4 and NVFP4 releases. inferenZ uses the NVFP4 one (100 GB):
- the experts of layers 1-39 in NVFP4: numbers of 4 bits with a scale for every 16 of them, quantized after training;
- the experts of layers 40-47, which the quantized releases keep at full precision, and everything else in BF16.

inferenZ converts them once into its own file: the NVFP4 experts exactly as published, the rest at 8 bits ([QUANTIZATION.md](QUANTIZATION.md)).
