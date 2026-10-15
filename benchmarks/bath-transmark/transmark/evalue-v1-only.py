#!/usr/bin/env python3
"""E-value distribution for hits found by one tool but not the other.

Joins two transmark ``*.out`` files on the embedded matching-model id
(``<family>/<digits>``) and keeps the smallest same-strand true-positive
E-value per id (same rule as ``evalue-scatter.py`` / ``bath-shared/
hitcounts.py``). Reports the E-value distribution of the ids unique to
the first (BATH) file, i.e. identified there but not in the second
(DUMMEST) file.

Note: both searches run with ``-E 10`` (``x-bathsearch``), so the right
edge of the distribution is a search cutoff, not a natural tail.

Usage:
    python3 evalue-v1-only.py <bath.out> <dummer.out> \
        [-t v1|v2|both] [--tsv out.tsv] [--plot out.png]
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bath-shared"),
)
from hitcounts import is_true_positive  # noqa: E402

FLOOR = 1e-300

DEFAULT_BATH = "transmark-full-fs-results/transmark-full-fs.out"
DEFAULT_DUMMER = "transmark-full-fs-results-v2/transmark-full-fs.out"


def parse_best_evalues(path: str) -> dict:
    """Parse a transmark ``.out`` -> {matching_model_id: best TP E-value}."""
    best: dict[str, float] = {}
    n_rows = n_tp = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            n_rows += 1
            try:
                e = float(parts[0])
            except ValueError:
                e = float("nan")
            fam = parts[5]
            hit = parts[-2]
            strand = parts[-1]
            if not is_true_positive(fam, hit, strand):
                continue
            n_tp += 1
            prev = best.get(hit)
            if prev is None:
                best[hit] = e
            elif math.isnan(e):
                continue
            elif math.isnan(prev) or e < prev:
                best[hit] = e
    print(f"{path}: rows={n_rows} tp_rows={n_tp} tp_hits={len(best)}")
    return best


def report(label: str, evals: dict, tsv: str, plot: str) -> None:
    finite = np.array([e for e in evals.values()
                       if not math.isnan(e)], dtype=float)
    n_zero = int((finite == 0).sum())
    log_e = np.log10(np.clip(finite, FLOOR, None))

    with open(tsv, "w") as f:
        f.write("hit\tbest_E\n")
        for h in sorted(evals):
            f.write(f"{h}\t{evals[h]:.6g}\n")

    print(f"{label}: finite N={len(finite)} nan={len(evals) - len(finite)} "
          f"E==0 underflow={n_zero}")
    qs = [0, 25, 50, 75, 100]
    for q, v in zip(qs, np.percentile(log_e, qs)):
        print(f"  log10E p{q:02d} = {v:.2f}  (E={10 ** v:.3g})")
    for t in (1e-10, 1e-5, 1e-3, 1.0, 10.0):
        print(f"  frac E<={t:g}: {float((finite <= t).mean()):.3f} "
              f"({int((finite <= t).sum())})")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    bins = np.linspace(log_e.min(), max(log_e.max(), 1.0), 60)
    ax1.hist(log_e, bins=bins, color="#3b6ea5", edgecolor="white", linewidth=0.3)
    ax1.axvline(math.log10(1e-5), color="red", ls="--", lw=1, label="E=1e-5")
    ax1.axvline(math.log10(1e-3), color="orange", ls="--", lw=1, label="E=1e-3")
    ax1.set_xlabel("log10(best E-value, BATH)")
    ax1.set_ylabel("hit count")
    ax1.set_title(f"{label} (N={len(finite)})")
    ax1.legend(frameon=True)

    order = np.sort(finite)
    cum = np.arange(1, len(order) + 1) / len(order)
    ax2.plot(np.clip(order, FLOOR, None), cum, color="#3b6ea5", lw=2)
    ax2.set_xscale("log")
    ax2.set_xlabel("E-value threshold T")
    ax2.set_ylabel("fraction of hits with E <= T")
    ax2.set_title("Cumulative E-value distribution")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.set_ylim(0, 1.02)

    plt.tight_layout()
    plt.savefig(plot, dpi=200)
    print(f"wrote {tsv}")
    print(f"wrote {plot}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bath_out", nargs="?", default=DEFAULT_BATH,
                    help="BATH transmark *.out")
    ap.add_argument("dummer_out", nargs="?", default=DEFAULT_DUMMER,
                    help="DUMMEST transmark *.out")
    ap.add_argument("-t", "--tool", choices=("v1", "v2", "both"), default="v1",
                    help="which unique set to report (default: v1)")
    ap.add_argument("--tsv", default="v1-only-evalues.tsv")
    ap.add_argument("--plot", default="v1-only-evalue-hist.png")
    args = ap.parse_args()

    bath = parse_best_evalues(args.bath_out)
    dummer = parse_best_evalues(args.dummer_out)
    inter = set(bath) & set(dummer)
    v1_only = {h: e for h, e in bath.items() if h not in dummer}
    v2_only = {h: e for h, e in dummer.items() if h not in bath}
    print(f"|bath|={len(bath)} |dummer|={len(dummer)} "
          f"|inter|={len(inter)} |v1_only|={len(v1_only)} |v2_only|={len(v2_only)}")

    if args.tool in ("v1", "both"):
        report("BATH-only TP hits", v1_only, args.tsv, args.plot)
    if args.tool in ("v2", "both"):
        tsv = args.tsv if args.tool == "v1" else args.tsv.replace("v1", "v2")
        plot = args.plot if args.tool == "v1" else args.plot.replace("v1", "v2")
        report("DUMMEST-only TP hits", v2_only, tsv, plot)


if __name__ == "__main__":
    main()
