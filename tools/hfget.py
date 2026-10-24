#!/usr/bin/env python3
"""hfget: priority-ordered, resumable download of a Hugging Face safetensors checkpoint.

Each remote shard is mirrored byte-for-byte into a sparse local file of the same size.
Tensors are fetched with HTTP range requests in priority order (dense weights first,
then routed experts layer by layer), so the converter can start on the dense part long
before the whole checkpoint has arrived. Completed ranges are appended to a journal,
so an interrupted download resumes where it stopped.

    python3 -I tools/hfget.py Qwen/Qwen3.8-Flash-Next-FP8 models/Qwen3.8-Flash-Next-FP8 [--workers 4] [--skip-vision]
"""
import argparse
import json
import os
import re
import struct
import subprocess
import sys
import threading
import time

CHUNK = 32 << 20


def curl(url, rng=None, retries=20):
    cmd = ["curl", "-sfL", "--retry", "10", "--retry-all-errors", "--connect-timeout", "30",
           "--speed-limit", "10000", "--speed-time", "60", "-A", "inferenz-hfget/1"]
    if rng:
        cmd += ["-r", rng]
    cmd.append(url)
    for attempt in range(retries):
        p = subprocess.run(cmd, capture_output=True)
        if p.returncode == 0:
            return p.stdout
        time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"curl failed for {url} {rng}: {p.stderr.decode(errors='replace')}")


def priority(name):
    # (class, order): smaller first. Dense weights, then routed experts layer by layer, then the rest that is
    # only read sparsely at runtime (MiniMax-M3.1-Flash and Qwen3.8-Flash-Next naming).
    if "vision" in name or "visual" in name or "multi_modal" in name or "patch_merge" in name:
        return (9, 0)
    m = re.search(r"layers\.(\d+)\.(?:block_sparse_moe|mlp)\.experts\.", name)
    if m:
        return (6, 0) if name.startswith("mtp.") else (5, int(m.group(1)))
    if "ngram_embedding" in name:
        return (4, 0)  # n-gram table: a few rows per token at runtime, but every forward pass needs it
    if ".dspark." in name or name.startswith("mtp."):
        return (2, 0)
    if "embed_tokens" in name:
        return (3, 0)  # only one row per token is needed at runtime; fetch after the hot dense set
    return (0, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-vision", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    base = f"https://huggingface.co/{a.repo}/resolve/main/"
    log = open(os.path.join(a.out, ".hfget.log"), "a", buffering=1)

    def say(msg):
        line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
        print(line, flush=True)
        log.write(line + "\n")

    idx_path = os.path.join(a.out, "model.safetensors.index.json")
    if not os.path.exists(idx_path):
        open(idx_path, "wb").write(curl(base + "model.safetensors.index.json"))
    files = sorted(set(json.load(open(idx_path))["weight_map"].values()))

    # headers (cached locally)
    hdr_dir = os.path.join(a.out, ".headers")
    os.makedirs(hdr_dir, exist_ok=True)
    headers = {}
    for fn in files:
        hp = os.path.join(hdr_dir, fn + ".raw")
        if not os.path.exists(hp):
            b = curl(base + fn, "0-7")
            n = struct.unpack("<Q", b[:8])[0]
            raw = curl(base + fn, f"8-{8 + n - 1}")
            assert len(raw) == n
            open(hp + ".tmp", "wb").write(raw)
            os.replace(hp + ".tmp", hp)
        raw = open(hp, "rb").read()
        headers[fn] = {"n": len(raw), "raw": raw, "header": json.loads(raw)}
    say(f"{len(files)} shards, headers ready")

    # sparse local files with the header in place
    jobs = []
    for fn in files:
        n = headers[fn]["n"]
        h = headers[fn]["header"]
        size = 8 + n + max((v["data_offsets"][1] for k, v in h.items() if k != "__metadata__"), default=0)
        path = os.path.join(a.out, fn)
        if not os.path.exists(path) or os.path.getsize(path) != size:
            with open(path, "wb") as f:
                f.truncate(size)
                f.seek(0)
                f.write(struct.pack("<Q", n))
                f.write(headers[fn]["raw"])
        for k, v in h.items():
            if k == "__metadata__":
                continue
            pr = priority(k)
            if a.skip_vision and pr[0] == 9:
                continue
            s, e = v["data_offsets"]
            jobs.append((pr, fn, 8 + n + s, 8 + n + e))

    # merge adjacent ranges of the same file and priority into chunks
    jobs.sort(key=lambda j: (j[0], j[1], j[2]))
    merged = []
    for pr, fn, s, e in jobs:
        if merged and merged[-1][0] == pr and merged[-1][1] == fn and merged[-1][3] == s and e - merged[-1][2] <= CHUNK:
            merged[-1][3] = e
        else:
            merged.append([pr, fn, s, e])
    chunks = []
    for pr, fn, s, e in merged:
        while s < e:
            t = min(e, s + CHUNK)
            chunks.append((pr, fn, s, t))
            s = t

    journal_path = os.path.join(a.out, ".hfget.done")
    done = set()
    if os.path.exists(journal_path):
        for line in open(journal_path):
            p = line.split()
            if len(p) == 3:
                done.add((p[0], int(p[1]), int(p[2])))
    todo = [c for c in chunks if (c[1], c[2], c[3]) not in done]
    total = sum(c[3] - c[2] for c in chunks)
    left = sum(c[3] - c[2] for c in todo)
    say(f"{len(chunks)} chunks, {total / 1e9:.2f} GB total, {left / 1e9:.2f} GB to fetch")

    journal = open(journal_path, "a", buffering=1)
    lock = threading.Lock()
    fds = {fn: os.open(os.path.join(a.out, fn), os.O_WRONLY) for fn in files}
    state = {"i": 0, "bytes": 0, "t0": time.time(), "last": 0.0, "class": None}

    def worker():
        while True:
            with lock:
                if state["i"] >= len(todo):
                    return
                pr, fn, s, e = todo[state["i"]]
                state["i"] += 1
            data = curl(base + fn, f"{s}-{e - 1}")
            if len(data) != e - s:
                with lock:
                    todo.append((pr, fn, s, e))
                continue
            os.pwrite(fds[fn], data, s)
            with lock:
                journal.write(f"{fn} {s} {e}\n")
                state["bytes"] += e - s
                now = time.time()
                if now - state["last"] > 60 or state["class"] != pr:
                    state["last"] = now
                    state["class"] = pr
                    rate = state["bytes"] / max(1e-9, now - state["t0"])
                    rem = left - state["bytes"]
                    say(f"class={pr} fetched {state['bytes'] / 1e9:.2f}/{left / 1e9:.2f} GB  "
                        f"{rate / 1e6:.2f} MB/s  eta {rem / max(rate, 1) / 3600:.1f} h")

    th = [threading.Thread(target=worker, daemon=True) for _ in range(a.workers)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    for fd in fds.values():
        os.fsync(fd)
        os.close(fd)
    say("complete")


if __name__ == "__main__":
    sys.exit(main())
