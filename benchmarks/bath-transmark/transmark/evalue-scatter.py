#!/usr/bin/env python3
"""2D scatter comparing per-hit best E-values: BATH vs DUMMEST.

Joins two transmark ``*.out`` files on the embedded-sequence id, keeping the
smallest E-value per embedded sequence per tool within each hit category.
Embedded sequences comprise the embedded test sequences and the planted
decoy sequences; background regions have no locus identity. True positives
are same-strand same-family overlaps with a test sequence; false positives
are split into decoy hits, background hits and wrong-target hits. Each
category is deduplicated independently (one point per sequence per
category), keeping the smallest E-value, to avoid inflating counts from
multiple fragmented alignments to the same locus. Background hits have no
per-locus identity and are hidden from the overlay and from all counts.
NaN pairs are excluded before computing Spearman r on log10 E-values.
``E=0`` underflows are clipped to 1e-300 for the log axis.

Figure text uses tool aliases only (BATH, DUMMEST).

Usage:
    python3 evalue-scatter.py <bath.out> <dummer.out> [-o out.png] [--no-fp]
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bath-shared"),
)
from hitcounts import (  # noqa: E402
    is_background_hit,
    is_planted_decoy,
    is_true_positive,
)

FLOOR = 1e-300

X_LABEL = "BATH"
Y_LABEL = "DUMMEST"


def parse_best_evalues(path: str, tp_only: bool = True) -> dict:
    """Parse a transmark .out file -> {embedded-sequence id: best evalue}.

    With ``tp_only=True`` keeps same-strand same-family true positives;
    with ``tp_only=False`` keeps the false-positive pool (decoy hits +
    background hits + wrong-target hits: every fragment where
    ``is_true_positive`` is False). Callers split the FP pool per category;
    each category is deduplicated per embedded-sequence id below, keeping
    the smallest E-value, via the shared ``is_true_positive`` rule
    (transmark-coverage.pl / -mer.pl).
    """
    best: dict[str, float] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                e = float(parts[0])
            except ValueError:
                e = float("nan")
            fam = parts[5]
            hit = parts[-2]
            strand = parts[-1]
            if is_true_positive(fam, hit, strand) != tp_only:
                continue
            prev = best.get(hit)
            if prev is None:
                best[hit] = e
            elif math.isnan(e):
                continue  # finite best always beats NaN
            elif math.isnan(prev) or e < prev:
                best[hit] = e
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bath_out", help="BATH transmark *.out")
    ap.add_argument("dummer_out", help="DUMMEST transmark *.out")
    ap.add_argument("-o", "--output", default="evalue-scatter.png")
    ap.add_argument("--no-fp", action="store_true",
                    help="Hide the false-positive overlay (TP-only legacy plot).")
    args = ap.parse_args()

    for p in (args.bath_out, args.dummer_out):
        if not os.path.isfile(p):
            raise SystemExit(f"missing input {p}")

    bath = parse_best_evalues(args.bath_out)
    dummer = parse_best_evalues(args.dummer_out)
    bath_fp = {} if args.no_fp else parse_best_evalues(
        args.bath_out, tp_only=False)
    dummer_fp = {} if args.no_fp else parse_best_evalues(
        args.dummer_out, tp_only=False)

    # Note: NaN pairs must be excluded explicitly -- scipy's spearmanr
    # defaults to nan_policy='propagate' and would return NaN otherwise.
    finite = [(bath[h], dummer[h]) for h in set(bath) & set(dummer)
              if not math.isnan(bath[h]) and not math.isnan(dummer[h])]

    if not finite:
        raise SystemExit("no finite overlapping pairs")

    # False positives shared by both tools: planted decoy hits in red,
    # wrong-target hits (wrong family and/or opposite strand) in yellow.
    # Each category is deduplicated per embedded sequence independently
    # (one point per sequence per category). Background hits have no
    # per-locus identity and are hidden from the overlay and all counts.
    n_fp_background_hidden = sum(
        1 for h in set(bath_fp) & set(dummer_fp)
        if is_background_hit(h)
        and not math.isnan(bath_fp[h]) and not math.isnan(dummer_fp[h]))
    fp_decoy = [(bath_fp[h], dummer_fp[h])
                for h in set(bath_fp) & set(dummer_fp)
                if is_planted_decoy(h)
                and not math.isnan(bath_fp[h])
                and not math.isnan(dummer_fp[h])]
    fp_nondecoy = [(bath_fp[h], dummer_fp[h])
                   for h in set(bath_fp) & set(dummer_fp)
                   if not is_planted_decoy(h) and not is_background_hit(h)
                   and not math.isnan(bath_fp[h])
                   and not math.isnan(dummer_fp[h])]

    bx = np.clip(np.array([a for a, _ in finite], dtype=float), FLOOR, None)
    dy = np.clip(np.array([b for _, b in finite], dtype=float), FLOOR, None)
    lx, ly = np.log10(bx), np.log10(dy)
    r, _ = spearmanr(lx, ly)

    print(f"{X_LABEL} vs {Y_LABEL}: N={len(finite)}, Spearman r (log10 E)={float(r):.3f}")
    if not args.no_fp:
        print(f"FP shared decoy hits (non-target)={len(fp_decoy)}, FP shared wrong-target={len(fp_nondecoy)}")

    plt.style.use("seaborn-v0_8-whitegrid")
    _, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(bx, dy, s=10, alpha=0.35, linewidths=0, rasterized=True,
               label=f"TP (N={len(finite)})")
    if fp_nondecoy:
        fx = np.clip(np.array([a for a, _ in fp_nondecoy], dtype=float),
                     FLOOR, None)
        fy = np.clip(np.array([b for _, b in fp_nondecoy], dtype=float),
                     FLOOR, None)
        ax.scatter(fx, fy, s=16, alpha=0.8, linewidths=0, color="yellow",
                   rasterized=True,
                   label=f"FP shared wrong-target (N={len(fp_nondecoy)})",
                   zorder=3)
    if fp_decoy:
        fx = np.clip(np.array([a for a, _ in fp_decoy], dtype=float),
                     FLOOR, None)
        fy = np.clip(np.array([b for _, b in fp_decoy], dtype=float),
                     FLOOR, None)
        ax.scatter(fx, fy, s=16, alpha=0.8, linewidths=0, color="red",
                   rasterized=True, label=f"FP shared decoy hits (N={len(fp_decoy)})",
                   zorder=4)

    hi = math.ceil(max(float(np.max(lx)), float(np.max(ly)))) + 0.5
    lims = (FLOOR * 0.9, 10 ** hi)
    diag = np.logspace(math.log10(lims[0]), math.log10(lims[1]), 100)
    ax.plot(diag, diag, linestyle="--", linewidth=1.2, color="black", alpha=0.7,
            label="y = x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_xlabel(f"{X_LABEL} lowest E-value per embedded sequence", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"{Y_LABEL} lowest E-value per embedded sequence", fontsize=12, fontweight="bold")
    ax.set_title(f"{X_LABEL} vs {Y_LABEL} E-values",
                 fontsize=13)
    ax.text(0.03, 0.97, f"N={len(finite)}\nSpearman r={float(r):.3f}",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    ax.legend(frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
