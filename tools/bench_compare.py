#!/usr/bin/env python3
"""Speed benchmark of a running inferenZ server, identical for every model and branch.

A warm-up pass (prompts that are not measured) lets the expert cache adapt; then every measured prompt runs greedy,
without thinking, with and without MTP speculative decoding. A long prompt measures prefill. The server's own
timings are recorded, plus a /v1/diag snapshot (cache hit rate, memory).

    python3 -I tools/bench_compare.py --url http://127.0.0.1:8080 --label nemotron --out bench-nemotron.json
"""
import argparse
import glob
import json
import os
import sys
import time
import urllib.request

WARMUP = [
    "Describe the water cycle in a few paragraphs.",
    "Scrivi una funzione Python che conta le vocali in una stringa e spiegala.",
    "What are the main differences between TCP and UDP?",
]

PROMPTS = [
    ("it_prose", "Scrivi un breve racconto (circa 300 parole) su un guardiano del faro in Sardegna durante una tempesta."),
    ("c_code", "Write a C function `int parse_ipv4(const char *s, uint32_t *out)` that parses a dotted-quad IPv4 "
               "address with full error checking (no leading zeros, each part 0-255, nothing after the address). "
               "Give the code and a short explanation."),
    ("js_code", "Implement `debounce(fn, wait, { leading, trailing })` in JavaScript, including `cancel()` and "
                "`flush()` methods, and explain the edge cases."),
    ("math", "A tank is filled by pipe A in 6 hours and by pipe B in 4 hours, and emptied by pipe C in 8 hours. "
             "All three are opened at 9:00 on an empty tank; at 10:30 pipe C is closed. At what time is the tank full? "
             "Show the reasoning step by step."),
    ("en_tech", "Explain step by step how inserting a key into a B-tree of order 4 works, including node splits "
                "that propagate to the root, with a small worked example."),
]


def post(url, body, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read())


def chat(url, prompt, max_tokens, spec):
    body = {"messages": [{"role": "user", "content": prompt}], "reasoning_effort": "none", "temperature": 0,
            "max_tokens": max_tokens, "speculative": spec, "stream": False}
    t0 = time.time()
    r = post(url + "/v1/chat/completions", body)
    wall = time.time() - t0
    t = r.get("timings", {})
    msg = r["choices"][0]["message"]
    return {"wall_s": wall, "prompt_n": t.get("prompt_n"), "prompt_cached": t.get("prompt_cached"),
            "prompt_tps": t.get("prompt_per_second"), "ttft_ms": t.get("ttft_ms"), "gen_n": t.get("predicted_n"),
            "decode_tps": t.get("predicted_per_second"), "drafted": t.get("drafted"), "accepted": t.get("accepted"),
            "finish": r["choices"][0].get("finish_reason"), "text": (msg.get("content") or "")}


def long_prompt(root):
    text = ""
    for f in sorted(glob.glob(os.path.join(root, "docs", "*.md"))):
        text += open(f, encoding="utf-8").read() + "\n\n"
    text = text[:14000]
    return ("Here is the documentation of a software project.\n\n" + text +
            "\n\nIn five bullet points, summarize what the project does and how it is organized.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    a = ap.parse_args()
    res = {"label": a.label, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": []}
    res["diag_start"] = get(a.url + "/v1/diag")
    for p in WARMUP:
        r = chat(a.url, p, 128, True)
        print(f"warm-up: {r['gen_n']} tokens @ {r['decode_tps']:.2f} tok/s", file=sys.stderr)
    for spec in (False, True):
        for name, p in PROMPTS:
            r = chat(a.url, p, a.tokens, spec)
            r.update(name=name, spec=spec)
            res["runs"].append(r)
            acc = f", MTP {r['accepted']}/{r['drafted']}" if r["drafted"] else ""
            print(f"{name:9s} spec={int(spec)}: prefill {r['prompt_n']} @ {r['prompt_tps']:.1f} tok/s, "
                  f"{r['gen_n']} gen @ {r['decode_tps']:.2f} tok/s{acc}", file=sys.stderr)
    r = chat(a.url, long_prompt(a.root), 64, True)
    r.update(name="long_prefill", spec=True)
    res["runs"].append(r)
    print(f"long prompt: {r['prompt_n']} tokens @ {r['prompt_tps']:.1f} tok/s, ttft {r['ttft_ms'] / 1e3:.1f} s", file=sys.stderr)
    res["diag_end"] = get(a.url + "/v1/diag")
    json.dump(res, open(a.out, "w"), indent=1)
    for spec in (False, True):
        rs = [r for r in res["runs"] if r["spec"] == spec and r["name"] != "long_prefill"]
        tot_n = sum(r["gen_n"] for r in rs)
        tot_t = sum(r["gen_n"] / r["decode_tps"] for r in rs if r["decode_tps"])
        print(f"{a.label}: decode {'MTP' if spec else 'plain'} {tot_n / tot_t:.2f} tok/s over {tot_n} tokens", file=sys.stderr)


if __name__ == "__main__":
    main()
