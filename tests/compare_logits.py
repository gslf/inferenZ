# Compare logits: reference (.npy from tests/ref_qwen38.py, or raw float32 [n][vocab] from inz dump) against
# engine logits (raw float32). Prints cosine, KL(ref || engine), top-1 and top-5 agreement.
#
#   python3 tests/compare_logits.py REF.npy|REF.f32 ENGINE.f32
import sys

import numpy as np

V = 248320


def load(path):
    if path.endswith(".npy"):
        return np.load(path).astype(np.float64)
    return np.fromfile(path, dtype=np.float32).reshape(-1, V).astype(np.float64)


ref = load(sys.argv[1])
eng = load(sys.argv[2])
n = min(len(ref), len(eng))
ref, eng = ref[-n:], eng[-n:]


def lsm(x):
    m = x.max(1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(1, keepdims=True))


kl = np.zeros(n)
for i in range(0, n, 128):  # in slices: [n][248K] float64 log-softmaxes are large
    lp, lq = lsm(ref[i:i + 128]), lsm(eng[i:i + 128])
    kl[i:i + 128] = (np.exp(lp) * (lp - lq)).sum(1)
cos = (ref * eng).sum(1) / np.linalg.norm(ref, axis=1) / np.linalg.norm(eng, axis=1)
top1 = (ref.argmax(1) == eng.argmax(1)).mean()
top5 = np.mean([len(set(np.argpartition(-ref[i], 5)[:5]) & set(np.argpartition(-eng[i], 5)[:5])) / 5 for i in range(n)])
print(f"positions {n}  cosine min {cos.min():.6f} mean {cos.mean():.6f}  KL mean {kl.mean():.5f} p95 {np.percentile(kl, 95):.5f} "
      f"max {kl.max():.5f}  top1 {top1:.3f}  top5-overlap {top5:.3f}")
