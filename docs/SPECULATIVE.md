# Speculative decoding

[README](../README.md)

Laguna S 2.1 has no multi-token prediction head. Poolside's speculative decoding uses DFlash, a separate draft model (`poolside/Laguna-S-2.1-DFlash-NVFP4`) that proposes up to 15 tokens per step from the target's hidden states; inferenZ does not run it, so Laguna generates one token per step and `--spec-k` / `--no-spec` have no effect.

The verification machinery stays in the engine: every cache is positional and the kernels sum each row in the same order for one token or several, so a pass over several tokens followed by a rewind gives the same logits as plain decoding (`inz dump --snaptest` checks it).
