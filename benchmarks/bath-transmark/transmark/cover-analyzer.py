#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labels import pretty_label  # noqa: E402

from hitcounts import (  # noqa: E402
    find_transmark_hits,
    is_background_hit,
    is_planted_decoy,
    is_true_positive,
    parse_transmark_out,
)


def parse_file(filepath: str):
    """
    Parse whitespace-delimited files with columns like:

    # HIT  HIT-ALT  TRUE-LENGTH  COVERED  OVER  CONTIGS  TOOL

    NOTE: no decoy filtering here by design. transmark-coverage.pl only
    emits same-family/same-strand true positives (it drops decoy hits),
    so .cover inputs contain no decoy rows by construction. (Decoys still
    matter on the ROC side, where transmark-rocplot counts them as false
    positives -- that is outside this script.)
    """

    rows = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            # Skip header lines
            if "TRUE-LENGTH" in line or line.startswith("#"):
                continue

            # Split on any amount of whitespace
            parts = line.split()

            if len(parts) < 7:
                continue

            try:
                row = {
                    "hit": parts[0],
                    "hit_alt": parts[1],
                    "true_length": float(parts[2]),
                    "covered": float(parts[3]),
                    "over": float(parts[4]),
                    "contigs": parts[5],
                    "tool": parts[6],
                }

                row["percent"] = (
                    row["covered"] / row["true_length"] * 100
                    if row["true_length"] > 0
                    else 0
                )

                row["over_percent"] = (
                    row["over"] / row["true_length"] * 100
                    if row["true_length"] > 0
                    else 0
                )

                rows.append(row)

            except ValueError:
                continue

    return rows


# Output columns: (txt header, tex header, summary key, width, format).
# Hits are deduplicated by matching-hit id (best E per id) to avoid inflating
# counts from multiple fragmented alignments to the same embedded sequence.
# Planted decoy hits (``decoy/N``, N>0) have per-locus ids and are deduplicated;
# background hits (catch-all ``decoy/0``: HMM background / junk / random with
# no embedded locus) share one id for all loci, so they are reported as raw
# non-unique fragment counts in ``n_fp_background``.
COLUMNS = (
    ("TOOL", "Tool", "tool", 25, "{}"),
    ("TRUE_POSITIVE_HITS", "True-Positive hits", "n_tp_hits", 18, "{}"),
    ("FALSE_POSITIVE_DECOY", "False-Positive decoy", "n_fp_decoy", 21, "{}"),
    ("FALSE_POSITIVE_BACKGROUND", "False-Positive background",
     "n_fp_background", 27, "{}"),
    ("FALSE_POSITIVE_NON_DECOY", "False-Positive non-decoy", "n_fp_nondecoy",
     25, "{}"),
    ("SUM_COVERED", "Sum covered", "covered_sum", 15, "{:.2f}"),
    ("SUM_TRUE_LEN", "Sum true len", "true_length_sum", 15, "{:.2f}"),
    ("WEIGHTED_%", "Weighted cov (\\%)", "weighted_percent", 15, "{:.2f}"),
    ("MEAN_ROW_%", "Mean coverage (\\%)", "mean_row_percent", 15, "{:.2f}"),
    ("WEIGHTED_OVER_%", "Weighted over (\\%)", "weighted_over_percent", 15,
     "{:.2f}"),
    ("MEAN_OVER_%", "Mean overextension (\\%)", "mean_over_percent", 15,
     "{:.2f}"),
    ("MEAN_OVER", "Mean overextension", "mean_over", 15, "{:.2f}"),
)

# Subset (and order) used for the paper's LaTeX table.
LATEX_KEYS = ("tool", "n_tp_hits", "n_fp_decoy", "n_fp_background",
              "n_fp_nondecoy",
              "mean_row_percent", "mean_over_percent")
LATEX_KEYS_RAW = ("tool", "n_tp_hits", "n_fp_decoy", "n_fp_background",
                  "n_fp_nondecoy",
                  "mean_row_percent", "mean_over")


def render_text(summary, over_percent=False):
    skip = ("mean_over",) if over_percent else (
        "weighted_over_percent", "mean_over_percent")
    cols = tuple(c for c in COLUMNS if c[2] not in skip)
    header = "".join(
        f"{txt_h:<{w}}" if key == "tool" else f"{txt_h:>{w}}"
        for txt_h, _tex_h, key, w, _fmt in cols
    )
    print(header)
    print("-" * len(header))
    for tool in sorted(summary):
        s = summary[tool]
        s["tool"] = tool
        line = "".join(
            (f"{tool:<{w}}" if key == "tool" else f"{fmt.format(s[key]):>{w}}")
            for _txt_h, _tex_h, key, w, fmt in cols
        )
        print(line)


def render_latex(summary, over_percent=False):
    keys = LATEX_KEYS if over_percent else LATEX_KEYS_RAW
    cols = [c for c in COLUMNS if c[2] in keys]
    # Reorder to keys order.
    cols = sorted(cols, key=lambda c: keys.index(c[2]))
    spec = "l" + "c" * (len(cols) - 1)
    print("\\begin{table}")
    print("\\centering")
    print("\\caption{}")
    print("\\label{tab:coverage}")
    print(f"\\begin{{tabular}}{{{spec}}}")
    print("\\hline")
    print(" & ".join(tex_h for _t, tex_h, _k, _w, _f in cols) + " \\\\")
    print("\\hline")
    for tool in sorted(summary):
        s = summary[tool]
        cells = []
        for _txt_h, _tex_h, key, _w, fmt in cols:
            if key == "tool":
                cells.append(tool.replace("_", "\\_"))
            else:
                cells.append(fmt.format(s[key]))
        print(" & ".join(cells) + " \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input files or glob patterns (example: results/*.cover)",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Emit a LaTeX tabular instead of the text table.",
    )
    parser.add_argument(
        "--over-percent",
        action="store_true",
        help="Show overextension as 2 percentage columns instead of the "
             "default single raw mean-OVER column.",
    )
    args = parser.parse_args()

    files = []
    for pattern in args.inputs:
        matches = glob.glob(pattern)
        if matches:
            files.extend(matches)
        elif os.path.isfile(pattern):
            files.append(pattern)
    files = sorted(set(files))

    if not files:
        raise SystemExit("No input files found.")

    tool_stats = defaultdict(
        lambda: {
            "covered_sum": 0.0,
            "true_length_sum": 0.0,
            "percents": [],
            "over_sum": 0.0,
            "over_percents": [],
        }
    )

    for filepath in files:
        rows = parse_file(filepath)
        # Group by the shared paper display name when the filename matches
        # (splits BATH vs BATH --no-fs, whose .cover files both stamp
        # TOOL=bathsearch).
        tool = pretty_label(filepath) or os.path.basename(filepath)
        for row in rows:
            tool_stats[tool]["covered_sum"] += row["covered"]
            tool_stats[tool]["true_length_sum"] += row["true_length"]
            tool_stats[tool]["percents"].append(row["percent"])
            tool_stats[tool]["over_sum"] += row["over"]
            tool_stats[tool]["over_percents"].append(row["over_percent"])

    # True-Positive hits: distinct ``<family>/<digits>`` matching-hit ids
    # from the sibling classified *.out file, keeping only same-family,
    # same-strand true positives (the rule shared with evalue-scatter.py /
    # bath-shared/hitcounts.py). The .out basename never identifies the
    # tool, so hits are attributed via the .cover file's display label.
    # False positives are split into planted decoy (``decoy/N``, N>0),
    # background (catch-all ``decoy/0``: no embedded locus, counted as raw
    # fragments) and non-decoy (wrong-family + opposite-strand) buckets,
    # mirroring evalue-scatter.py.
    tp_by_tool: dict[str, set] = defaultdict(set)
    decoy_fp_by_tool: dict[str, set] = defaultdict(set)
    nondecoy_fp_by_tool: dict[str, set] = defaultdict(set)
    background_fp_frags_by_tool: dict[str, int] = defaultdict(int)
    for filepath in files:
        tool = pretty_label(filepath) or os.path.basename(filepath)
        hits_path = find_transmark_hits(filepath)
        if hits_path is None:
            print(
                f"warning: no sibling *.out/*.out.tmp for {filepath}; "
                f"True-Positive hits=0 for {tool}",
                file=sys.stderr,
            )
            tp_by_tool[tool]
            decoy_fp_by_tool[tool]
            nondecoy_fp_by_tool[tool]
            background_fp_frags_by_tool[tool]
            continue
        hits, _pb, classified = parse_transmark_out(hits_path)
        for (_read, fam, _e, _s, _span), (mh, strand) in zip(hits, classified):
            if is_true_positive(fam, mh, strand):
                tp_by_tool[tool].add(mh)
            elif mh is not None:
                if is_background_hit(mh):
                    background_fp_frags_by_tool[tool] += 1
                elif is_planted_decoy(mh):
                    decoy_fp_by_tool[tool].add(mh)
                else:
                    nondecoy_fp_by_tool[tool].add(mh)

    summary = {}
    for tool in sorted(set(tool_stats) | set(tp_by_tool) | set(decoy_fp_by_tool)
                       | set(nondecoy_fp_by_tool)):
        stats = tool_stats[tool]
        n_rows = len(stats["percents"])
        true_length_sum = stats["true_length_sum"]
        summary[tool] = {
            "n_tp_hits": len(tp_by_tool.get(tool, ())),
            "n_fp_decoy": len(decoy_fp_by_tool.get(tool, ())),
            "n_fp_background": background_fp_frags_by_tool.get(tool, 0),
            "n_fp_nondecoy": len(nondecoy_fp_by_tool.get(tool, ())),
            "covered_sum": stats["covered_sum"],
            "true_length_sum": true_length_sum,
            "weighted_percent": (
                stats["covered_sum"] / true_length_sum * 100
                if true_length_sum > 0 else 0
            ),
            "mean_row_percent": (
                sum(stats["percents"]) / n_rows if n_rows > 0 else 0
            ),
            "weighted_over_percent": (
                stats["over_sum"] / true_length_sum * 100
                if true_length_sum > 0 else 0
            ),
            "mean_over_percent": (
                sum(stats["over_percents"]) / n_rows if n_rows > 0 else 0
            ),
            "mean_over": (
                stats["over_sum"] / n_rows if n_rows > 0 else 0
            ),
        }

    if args.latex:
        render_latex(summary, args.over_percent)
    else:
        render_text(summary, args.over_percent)


if __name__ == "__main__":
    main()
