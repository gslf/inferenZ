#!/usr/bin/env python3
"""Side-by-side table of tools/bench_compare.py results (and optionally tests/code_eval.py logs).

    python3 -I tools/bench_report.py A.json B.json [--eval A.log[,A2.log] B.log[,B2.log]]
"""
import argparse
import json
import re


def decode(res, spec):
    rs = [r for r in res["runs"] if r["spec"] == spec and r["name"] != "long_prefill"]
    n = sum(r["gen_n"] for r in rs)
    t = sum(r["gen_n"] / r["decode_tps"] for r in rs if r["decode_tps"])
    return n / t if t else 0


def acceptance(res):
    rs = [r for r in res["runs"] if r["spec"] and r["drafted"]]
    d = sum(r["drafted"] for r in rs)
    return sum(r["accepted"] for r in rs) / d if d else 0


def short_prefill(res):
    rs = [r for r in res["runs"] if r["name"] != "long_prefill"]
    n = sum(r["prompt_n"] - (r["prompt_cached"] or 0) for r in rs)
    t = sum((r["prompt_n"] - (r["prompt_cached"] or 0)) / r["prompt_tps"] for r in rs if r["prompt_tps"])
    return n / t if t else 0


def eval_rate(paths):
    p = t = 0
    for path in paths.split(","):
        for line in open(path):
            if line.startswith(("PASS ", "FAIL ")):
                t += 1
                p += line.startswith("PASS")
    return p, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--eval", nargs=2)
    a = ap.parse_args()
    A, B = json.load(open(a.a)), json.load(open(a.b))
    rows = []
    for name, f in [("decode, plain (tok/s)", lambda r: f"{decode(r, False):.2f}"),
                    ("decode, MTP (tok/s)", lambda r: f"{decode(r, True):.2f}"),
                    ("MTP acceptance", lambda r: f"{100 * acceptance(r):.0f}%"),
                    ("prefill, short prompts (tok/s)", lambda r: f"{short_prefill(r):.1f}"),
                    ("prefill, long prompt (tok/s)", lambda r: f"{[x for x in r['runs'] if x['name'] == 'long_prefill'][0]['prompt_tps']:.1f}"),
                    ("long prompt tokens", lambda r: f"{[x for x in r['runs'] if x['name'] == 'long_prefill'][0]['prompt_n']}"),
                    ("TTFT, long prompt (s)", lambda r: f"{[x for x in r['runs'] if x['name'] == 'long_prefill'][0]['ttft_ms'] / 1e3:.1f}"),
                    ("expert cache hit rate", lambda r: f"{100 * r['diag_end']['cache']['hit_rate']:.1f}%"),
                    ("RSS (GB)", lambda r: f"{r['diag_end']['memory']['rss_gb']:.1f}"),
                    ("dense weights in RAM (GB)", lambda r: f"{r['diag_end']['memory']['dense_gb']:.1f}"),
                    ("expert cache (GB)", lambda r: f"{r['diag_end']['memory']['cache_gb']:.1f}")]:
        try:
            rows.append((name, f(A), f(B)))
        except (KeyError, IndexError, TypeError, ZeroDivisionError):
            rows.append((name, "?", "?"))
    if a.eval:
        (pa, ta), (pb, tb) = eval_rate(a.eval[0]), eval_rate(a.eval[1])
        rows.append(("code_eval pass rate", f"{pa}/{ta} ({100 * pa / max(ta, 1):.0f}%)", f"{pb}/{tb} ({100 * pb / max(tb, 1):.0f}%)"))
    print(f"| | {A['label']} | {B['label']} |\n|---|---:|---:|")
    for r in rows:
        print(f"| {r[0]} | {r[1]} | {r[2]} |")
    print()
    print(f"| per prompt (decode tok/s, plain / MTP) | {A['label']} | {B['label']} |\n|---|---:|---:|")
    for name in [r["name"] for r in A["runs"] if not r["spec"] and r["name"] != "long_prefill"]:
        def cell(res):
            p = [r for r in res["runs"] if r["name"] == name]
            return " / ".join(f"{r['decode_tps']:.2f}" for r in sorted(p, key=lambda r: r["spec"]))
        print(f"| {name} | {cell(A)} | {cell(B)} |")


if __name__ == "__main__":
    main()
