# Compare the engine's residual stream after every layer (INZ_DUMP_HIDDEN, float32 [L][T][10240]) with the
# reference's (tests/ref_qwen38.py --dump-hidden, same layout). Prints, per layer, the relative L2 error per
# token (mean and worst) and the cosine of the worst token: a bug shows up as a jump at the layer that has it.
#
#   python3 tests/compare_hidden.py ref_hidden.f32 eng_hidden.f32 T [CHUNK | n1,n2,...]
import sys

import numpy as np

W = 10240
ref = np.fromfile(sys.argv[1], dtype=np.float32)
eng = np.fromfile(sys.argv[2], dtype=np.float32)
T = int(sys.argv[3])
# the engine dumps prefill chunk by chunk ([chunk][L][n][W]): CHUNK is a chunk size, or a comma-separated list of
# sizes (inz dump --last N splits the prompt before the first logits row)
sizes = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [T]
L = min(ref.size, eng.size) // (T * W)
ref = ref[: L * T * W].reshape(L, T, W).astype(np.float64)
parts, off, t0, i = [], 0, 0, 0
while t0 < T:
    n = min(sizes[min(i, len(sizes) - 1)], T - t0)
    parts.append(eng[off: off + L * n * W].reshape(L, n, W))
    off += L * n * W
    t0 += n
    i += 1
eng = np.concatenate(parts, axis=1).astype(np.float64)
print(f"{L} layers x {T} tokens")
for l in range(L):
    r, e = ref[l], eng[l]
    rel = np.linalg.norm(e - r, axis=1) / np.maximum(np.linalg.norm(r, axis=1), 1e-30)
    w = int(rel.argmax())
    cos = float((r[w] * e[w]).sum() / max(np.linalg.norm(r[w]) * np.linalg.norm(e[w]), 1e-30))
    print(f"layer {l:2d}  rel err mean {rel.mean():.2e}  max {rel.max():.2e} (token {w}, cos {cos:.6f})  |x| {np.linalg.norm(r, axis=1).mean():.1f}")
