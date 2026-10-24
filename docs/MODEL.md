# Qwen3.8-Flash-Next

[README](../README.md)

The model has 125 billion parameters, but each token uses only about 6 billion of them. Most of the weights are *experts* that are consulted only when they are relevant. That is what lets a model this large run on a laptop: for each token, the engine only needs the small part of the model that the token uses.

A token goes through 48 layers. Each layer does two things:
1. it looks at the earlier tokens, to bring in context;
2. it transforms what it has gathered, with a few experts chosen for this token.

At the end the model gives a score to each of the 248,320 tokens of its vocabulary, and one of them is picked.

## Tokens in, tokens out

A tokenizer splits the text into tokens. Each token starts as a list of 2,560 numbers looked up in a table (the *embedding*). After the last layer, the *LM head* turns the final numbers back into a score for every token of the vocabulary. The scores become probabilities, and the next token is drawn from them.

## Four parallel streams

In most models, each layer reads one running list of numbers (the *residual stream*), computes something, and adds its result back. Qwen3.8-Flash-Next keeps four of these streams side by side (*hyper-connections*).
- Before each block, a small learned network decides how to blend the four streams into the block's input.
- After the block, it decides how much of the result to add to each stream.

The layers thus have more freedom in what they pass on to later layers, at a small cost. At the end, the four streams are blended into one.

## Two ways of looking at the past

To predict the next token the model must use what came before. Classic attention compares the current token with every earlier one. It´s precise, but its cost grows with the length of the text. This model combines two cheaper mechanisms.

### Gated DeltaNet: a compact memory (36 layers)

Three layers out of four keep a memory of fixed size instead of looking back at the tokens. The memory holds associations between *keys* (what something is about) and *values* (what to remember about it). For each new token:
- the memory fades a little, more or less depending on the token (the *gate*), old details give way to new ones;
- the model checks what the memory already says about the current key and writes in only the difference (the *delta rule*), so the same thing is not stored twice;
- the token reads from the memory what it needs (its *query*).

The cost per token is the same at token 10 as at token 100,000. The price is that the memory is a summary: it cannot quote an earlier passage word for word.

### QSA: sparse attention (12 layers)

Every fourth layer uses real attention, which can retrieve exact earlier tokens. To keep it cheap, it does not read all of them. A small *indexer* groups the text into blocks of 4 tokens and scores how relevant each block is to the current token. Attention then reads only the 512 best blocks (2,048 tokens), plus the latest unfinished block. 

Together, the DeltaNet layers carry the gist of a long text cheaply, and the QSA layers fetch exact details when they are needed.

## Mixture of experts

After looking at the context, each layer transforms the token with a feed-forward network. Instead of one large network, each layer has 512 small ones, the *experts*. A *router* picks the 10 best suited to the token and weights their answers. One more expert, the *shared expert*, is always used, for what every token needs.

The experts hold 121 of the 125 billion parameters, but a token uses only 10 of the 512 in each layer. This is why inferenZ can keep the experts on the SSD and only the ones in use in RAM ([STREAMING.md](STREAMING.md)).

## The n-gram table

Before its second layer, the model consults a very large table (320 million rows, 51 GB), indexed by the last two or three tokens. It works like a dictionary of short token sequences: what the model knows about common combinations is looked up instead of being computed. Any combination has a row, because the table is indexed by a hash of the tokens. A gate decides from the context how much of what was looked up to use. The lookup restarts at the end of a document, so it never mixes two unrelated texts.

The table is huge, but each token reads only 16 small rows of it. inferenZ therefore leaves it on the SSD and lets the operating system keep the frequent rows in memory.

## MTP (speculative decoding)

The model comes with one extra layer, trained to guess the token after the next one (*multi-token prediction*). inferenZ uses it to draft a token cheaply. The main model then checks the draft and produces its own next token in a single pass. When the draft is right (over 90% of the time in our tests), two tokens come out for little more than the cost of one. The output is exactly what the model would have written without the draft ([SPECULATIVE.md](SPECULATIVE.md)).

## In numbers

| | |
|---|---|
| parameters | 125 B, 121 B of them in experts; about 6 B used per token |
| plus | the n-gram table (51 B entries) and the MTP layer (4 B) |
| layers | 48: 36 Gated DeltaNet and 12 QSA attention, plus the MTP layer |
| numbers per token | 2,560, in 4 parallel streams |
| experts | 512 per layer, 10 chosen per token, plus 1 shared |
| vocabulary | 248,320 tokens |

## The official weights

Qwen publishes the model as 184 GB of weights:
- the experts and the n-gram table in FP8, numbers of 8 bits, with a scale factor for each block of weights;
- everything else in BF16, numbers of 16 bits.

inferenZ converts them once into its own file, choosing a format for each kind of tensor ([QUANTIZATION.md](QUANTIZATION.md)).
