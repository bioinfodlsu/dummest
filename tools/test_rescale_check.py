#!/usr/bin/env python3
"""Assertion checker for test_rescale.sh output.

Usage: test_rescale_check.py FILE EXP_SCORE TOL EXP_E EXP_ANCHOR EXP_SPANS FP_EVALUE

Fingerprints the true (top) hit against the expected score/E/anchor/spans,
requires that true hit to be significant (E < FP_EVALUE), and flags any other
hit whose E is significant (E < FP_EVALUE) as a false positive.

EXP_E is a full-match regex; EXP_ANCHOR or EXP_SPANS may be '-' to skip.
"""
import re
import sys


def e2f(s):
    try:
        return float(s)
    except ValueError:
        return float("inf")


def main():
    if len(sys.argv) != 8:
        sys.exit(f"usage: {sys.argv[0]} FILE EXP_SCORE TOL EXP_E EXP_ANCHOR EXP_SPANS FP_EVALUE")
    path, exp_score, tol, exp_e, exp_anchor, exp_spans, fp_s = sys.argv[1:8]
    exp_score, tol, fp = float(exp_score), float(tol), float(fp_s)

    txt = open(path).read()
    if re.search(r"nan|inf", txt, re.I):
        print(f"FAIL {path}: nan/inf token present")
        sys.exit(1)

    recs, cur = [], None
    for line in txt.splitlines():
        if line.startswith("a "):
            m = re.match(r"a score=(\S+) E=(\S+) anchor=(\S+)", line)
            cur = {"score": m.group(1), "e": m.group(2),
                   "anchor": m.group(3), "s": []}
            recs.append(cur)
        elif line.startswith("s ") and cur is not None:
            cur["s"].append(tuple(line.split()[1:5]))

    if not recs:
        print(f"FAIL {path}: no hits")
        sys.exit(1)

    top = recs[0]
    problems = []

    if abs(float(top["score"]) - exp_score) > tol:
        problems.append(f"true-hit score {top['score']} != {exp_score}+-{tol}")
    if not re.fullmatch(exp_e, top["e"]):
        problems.append(f"true-hit E {top['e']} !~ {exp_e}")
    if exp_anchor != "-" and top["anchor"] != exp_anchor:
        problems.append(f"true-hit anchor {top['anchor']} != {exp_anchor}")
    if exp_spans != "-":
        want = [tuple(x.split(",")) for x in exp_spans.split(";")]
        if top["s"] != want:
            problems.append(f"true-hit spans {top['s']} != {want}")
    if not e2f(top["e"]) < fp:
        problems.append(f"true-hit E {top['e']} not significant (<{fp:g})")

    false_hits = [(r["anchor"], r["e"]) for r in recs[1:] if e2f(r["e"]) < fp]
    if false_hits:
        problems.append(f"significant false hits (<{fp:g}): {false_hits}")

    if problems:
        for p in problems:
            print(f"FAIL {path}: {p}")
        sys.exit(1)

    others = len(recs) - 1
    print(f"ok {path}: true={top['anchor']} score={top['score']} E={top['e']} "
          f"spans={top['s']} | {others} non-significant other hit(s)")


if __name__ == "__main__":
    main()
