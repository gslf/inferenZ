# Speculative decoding with the MTP layer

[README](../README.md)

Decoding one token reads every dense weight and ten experts per layer. On a CPU that is a memory-bandwidth cost, nearly independent of how many tokens are checked at once. Qwen3.8-Flash-Next ships a multi-token-prediction (MTP) layer trained to guess the token after next. inferenZ uses it to draft tokens and verifies them with the main model in one batched pass.

## The MTP layer

One QSA attention + MoE decoder layer with its own hyper-connection mixer. It shares the embeddings and the LM head with the main model. Its input combines the main model's residual at position `p` with the embedding of the token at `p + 1`, and it predicts the token at `p + 2`:

```
X   = [fc_hidden · norm(X_main_p,s)]_s + fc_embedding · norm(embed[t_p+1])     (all 4 streams)
X   = mtp_layer(X);  logits = lm_head · mtp_mixer(X)
```

The MTP attention has its own cache, one slot per position. Slot `p` needs the token at `p + 1`, so the MTP cache always runs one position behind the main one:
- **prefill:** after each prompt chunk, the slots whose next token is known are filled in one batch;
- **decode:** each step fills the pending slots and drafts from the newest one in a single MTP call.

The MTP layer's experts live in the same SSD-backed cache as the others, as layer 48.

## One step

1. **Token.** `tok` is the token sampled at the previous step, at position P.
2. **Draft.** The MTP layer fills its pending slots (P − m … P − 1, the last one with `tok`) and drafts `d0`. With `--spec-k` 2 or 3, further MTP steps continue from its newest output to draft `d1` and `d2`.
3. **Verify.** One pass of the main model over `[tok, d0 … d(k−1)]`, at positions P … P + k, gives the logits of every row.
4. **Accept.** Speculative sampling decides, position by position, which drafts to keep.
5. **Rewind.** If a draft is rejected, the recurrent state goes back to the snapshot taken after the last accepted token.

### Acceptance keeps the output distribution exact

A draft `d ~ q` (the MTP distribution after the same temperature, top-k, top-p and penalties) is accepted with probability `min(1, p(d) / q(d))`, where `p` is the main model's distribution at that position. On rejection the token is drawn from `normalize(max(0, p − q))`. This is the standard speculative-sampling rule: the generated tokens follow `p` exactly, whatever the drafts were. With temperature 0 it reduces to "accept while the draft equals the main model's argmax".


## Options

| Option | Effect |
|---|---|
| `--no-spec` | no MTP: plain decoding (the MTP cache is not maintained either) |
| `--spec-k N` | drafts per step, 1 to 3 (default 1). Draft `i > 0` chains the MTP layer on its own output |
| request field `"speculative": false` | plain decoding for one request (the MTP cache is still maintained) |

The monitor shows tokens drafted and accepted, and the accepted tokens per verify step.
