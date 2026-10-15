#!/usr/bin/env python3
"""E-value-filtered coverage/overextension with optional common-set restriction.

Reuses the original pipeline pieces instead of reimplementing them:
  * ``transmark-coverage.pl`` -- per-hit COVERED / OVER / CONTIGS computation
    (invoked via ``perl`` on sorted, filtered ``*.out`` data, exactly as
    ``transmark-pp.sh`` does).
  * ``bath-shared/hitcounts.py`` -- ``is_true_positive`` rule and
    ``find_transmark_hits`` sibling lookup (same TP definition as the perl).
  * ``cover-analyzer.py`` -- ``parse_file`` plus ``render_text`` /
    ``render_latex`` table rendering (same columns as coverage-*.txt/.tex).
  * ``labels.pretty_label`` -- paper display names (BATH vs BATH --no-fs).

Typical Table-2 style run (full-over-fs test):

    python3 coverage-filtered.py --pos transmark-full-over-fs.pos \\
        --evalue 1e-5 --common \\
        transmark-full-over-fs-results/bath-fs-analysis.cover \\
        transmark-full-over-fs-results-no-fs/bath-fs-no-fs-analysis.cover \\
        transmark-full-over-fs-results-v2/dummer-fs-analysis.cover \\
        transmark-full-over-fs-results-last/last-fs-analysis.cover

Positional inputs mirror ``cover-analyzer.py`` (files or globs of ``*.cover``).
For each ``*.cover`` the sibling classified ``*.out`` is located with
``find_transmark_hits`` and *reused* -- no search tool is rerun. The ``*.out``
is filtered to ``E <= --evalue`` (inclusive, essential) and optionally to the
intersection of hit ids seen by *all* tools (``--common``), staged under
``--tmpdir`` (default ``/tmp``), sorted, and piped through the perl script.
The resulting filtered ``*.cover`` files are then summarized with the same
weighted/mean coverage and overextension columns as ``cover-analyzer.py``.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import math
import os
import shutil
import subprocess
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
# labels.py wires bath-shared/ onto sys.path as a side effect.
from labels import pretty_label  # noqa: E402

sys.path.insert(
    0, os.path.join(SCRIPT_DIR, "..", "bath-shared"),
)
from hitcounts import (  # noqa: E402
    find_transmark_hits,
    is_background_hit,
    is_planted_decoy,
    is_true_positive,
)


def _load_cover_analyzer():
    path = os.path.join(SCRIPT_DIR, "cover-analyzer.py")
    spec = importlib.util.spec_from_file_location("cover_analyzer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load cover-analyzer.py from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cover_analyzer = _load_cover_analyzer()

# Display label -> TOOL token stamped by transmark-coverage.pl
# (transmark-pp.sh passes bathsearch/dummer/last; both BATH variants stamp
# "bathsearch" and are disambiguated by filename via pretty_label).
TOOL_TOKENS = {
    "BATH": "bathsearch",
    "BATH --no-fs": "bathsearch",
    "DUMMEST": "dummer",
    "DUMMEST (more sensitive)": "dummer",
    "LAST": "last",
}


def detect_tool_token(cover_path: str, display: str) -> str:
    """Recover the perl TOOL token without reading the whole .cover."""
    if display in TOOL_TOKENS:
        return TOOL_TOKENS[display]
    try:
        with open(cover_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or "TRUE-LENGTH" in s:
                    continue
                parts = s.split()
                if len(parts) >= 7:
                    return parts[6]
                break
    except OSError:
        pass
    return "tool"


def read_filtered_fragments(out_path: str, evalue_max: float):
    """Read a classified *.out, keep same-strand TP fragments with E<=max.

    Returns ([(evalue, raw_line)], {hit_id: best_evalue},
    {planted_decoy_hit_id: best_evalue}, {nondecoy_fp_hit_id: best_evalue},
    n_background_frags, n_nondecoy_frags).
    Raw lines are preserved verbatim for the perl stage. False positives
    (``is_true_positive`` False) are split into planted decoy
    (``is_planted_decoy``), background (catch-all ``decoy/0``, counted as raw
    fragments only) and non-decoy (opposite-strand + wrong-family).
    Hits are deduplicated by matching-hit id (best E per id) to avoid
    inflating counts from fragmented alignments.
    """
    kept: list[tuple[float, str]] = []
    best: dict[str, float] = {}
    best_decoy_fp: dict[str, float] = {}
    best_nondecoy_fp: dict[str, float] = {}
    n_background_frags = 0
    n_nondecoy_frags = 0
    with open(out_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 7:
                continue
            try:
                e = float(parts[0])
            except ValueError:
                continue
            if not (e <= evalue_max):
                # Covers e > max and NaN (NaN comparison is False).
                continue
            fam = parts[5]
            mh = parts[-2]
            strand = parts[-1]
            if not is_true_positive(fam, mh, strand):
                if is_background_hit(mh):
                    n_background_frags += 1
                    continue
                if is_planted_decoy(mh):
                    prev = best_decoy_fp.get(mh)
                    if prev is None or e < prev:
                        best_decoy_fp[mh] = e
                else:
                    n_nondecoy_frags += 1
                    prev = best_nondecoy_fp.get(mh)
                    if prev is None or e < prev:
                        best_nondecoy_fp[mh] = e
                continue
            kept.append((e, line if line.endswith("\n") else line + "\n"))
            prev = best.get(mh)
            if prev is None or e < prev:
                best[mh] = e
    return kept, best, best_decoy_fp, best_nondecoy_fp, n_background_frags, n_nondecoy_frags


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("inputs", nargs="+",
                    help="Input .cover files or globs (like cover-analyzer.py). "
                         "Sibling *.out is found automatically and reused.")
    ap.add_argument("--pos", required=True, help="Transmark .pos file.")
    ap.add_argument("--evalue", type=float, default=1e-5,
                    help="Inclusive E-value cutoff E<=T (default 1e-5). "
                         "Always applied; this is the essential filter.")
    ap.add_argument("--common", action="store_true",
                    help="Restrict to the common set: hit ids with best E<=T "
                         "in ALL input tools. Without it, per-tool E-filtered "
                         "coverage is reported.")
    ap.add_argument("--latex", action="store_true",
                    help="Emit LaTeX tabular instead of the text table.")
    ap.add_argument("--over-percent", action="store_true",
                    help="Show overextension as 2 percentage columns instead "
                         "of the default single raw mean-OVER column.")
    ap.add_argument("--perl", default=os.path.join(SCRIPT_DIR,
                                                   "transmark-coverage.pl"),
                    help="Path to transmark-coverage.pl.")
    ap.add_argument("--tmpdir", default="/tmp",
                    help="Parent dir for staging filtered .out/.cover.")
    ap.add_argument("--keep-tmp", action="store_true",
                    help="Keep the staging dir under --tmpdir for debugging.")
    args = ap.parse_args()

    if not os.path.isfile(args.pos):
        raise SystemExit(f"missing --pos file {args.pos}")
    if not os.path.isfile(args.perl):
        raise SystemExit(f"missing perl script {args.perl}")
    if not (args.evalue >= 0) or math.isnan(args.evalue):
        raise SystemExit("--evalue must be a non-negative number")

    files: list[str] = []
    for pattern in args.inputs:
        matches = glob.glob(pattern)
        if matches:
            files.extend(matches)
        elif os.path.isfile(pattern):
            files.append(pattern)
    files = sorted(set(files))
    if not files:
        raise SystemExit("No input files found.")

    # Resolve each .cover -> (display label, sibling .out, perl TOOL token).
    targets: list[dict] = []
    for cover in files:
        display = pretty_label(cover) or os.path.basename(cover)
        hits_path = find_transmark_hits(cover)
        if hits_path is None:
            raise SystemExit(
                f"no sibling *.out/*.out.tmp for {cover}; "
                f"cannot reuse .out for {display}")
        targets.append({
            "cover": cover,
            "display": display,
            "hits": hits_path,
            "tool_token": detect_tool_token(cover, display),
        })

    tmp_root = tempfile.mkdtemp(prefix="coverflt_", dir=args.tmpdir)
    try:
        # Phase A: E-filter each .out (essential), track best-E per hit.
        per_tool_frags: dict[str, list] = {}
        per_tool_best: dict[str, dict] = {}
        per_tool_decoy_fp_best: dict[str, dict] = {}
        per_tool_nondecoy_fp_best: dict[str, dict] = {}
        per_tool_background_frags: dict[str, int] = {}
        per_tool_nondecoy_fp_frags: dict[str, int] = {}
        for t in targets:
            frags, best, best_decoy_fp, best_nondecoy_fp, \
                n_background_frags, n_nondecoy_frags = \
                read_filtered_fragments(t["hits"], args.evalue)
            per_tool_frags[t["display"]] = frags
            per_tool_best[t["display"]] = best
            per_tool_decoy_fp_best[t["display"]] = best_decoy_fp
            per_tool_nondecoy_fp_best[t["display"]] = best_nondecoy_fp
            per_tool_background_frags[t["display"]] = n_background_frags
            per_tool_nondecoy_fp_frags[t["display"]] = n_nondecoy_frags
            print(f"[coverage-filtered] {t['display']}: "
                  f"{len(frags)} fragments, {len(best)} hits, "
                  f"{len(best_decoy_fp)} decoy FP, "
                  f"{n_background_frags} background frags, "
                  f"{len(best_nondecoy_fp)} non-decoy FP "
                  f"({n_nondecoy_frags} frags) "
                  f"with E<={args.evalue:g} from {t['hits']}",
                  file=sys.stderr)

        # Phase B: optional common-set intersection.
        common: set | None = None
        common_decoy_fp: set | None = None
        common_nondecoy_fp: set | None = None
        if args.common:
            sets = [set(b) for b in per_tool_best.values()]
            common = set.intersection(*sets) if sets else set()
            decoy_sets = [set(b) for b in per_tool_decoy_fp_best.values()]
            common_decoy_fp = (set.intersection(*decoy_sets) if decoy_sets
                               else set())
            nondecoy_sets = [
                set(b) for b in per_tool_nondecoy_fp_best.values()]
            common_nondecoy_fp = (set.intersection(*nondecoy_sets)
                                  if nondecoy_sets else set())
            print(f"[coverage-filtered] common set: {len(common)} hits "
                  f"({len(common_decoy_fp)} decoy FP, "
                  f"{len(common_nondecoy_fp)} non-decoy FP) "
                  f"with E<={args.evalue:g} in all {len(sets)} tools",
                  file=sys.stderr)

        # Phase C: stage filtered .out + perl -> filtered .cover under /tmp.
        # One subdir per tool so a manual `cover-analyzer.py /tmp/.../*.cover`
        # still finds the right sibling filtered .out via find_transmark_hits.
        filtered_covers: dict[str, str] = {}
        for t in targets:
            display = t["display"]
            allowed = common if common is not None else set(
                per_tool_best[display])
            frags = []
            for e, raw in per_tool_frags[display]:
                parts = raw.split()
                mh = parts[-2]
                if mh in allowed:
                    frags.append((e, raw))
            frags.sort(key=lambda p: p[0])

            slug = os.path.basename(t["cover"])
            subdir = os.path.join(tmp_root, slug + ".d")
            os.makedirs(subdir, exist_ok=True)
            filt_out = os.path.join(
                subdir, os.path.basename(t["hits"]))
            filt_cover = os.path.join(subdir, slug)
            with open(filt_out, "w", encoding="utf-8") as f:
                for _e, raw in frags:
                    f.write(raw)
            with open(filt_out, "r", encoding="utf-8") as fin, open(
                    filt_cover, "w", encoding="utf-8") as fout:
                subprocess.run(
                    ["perl", args.perl, args.pos, t["tool_token"]],
                    stdin=fin, stdout=fout, check=True)
            filtered_covers[display] = filt_cover
            print(f"[coverage-filtered] {display}: staged "
                  f"{len(frags)} fragments -> {filt_cover} "
                  f"(TOOL={t['tool_token']})", file=sys.stderr)

        # Phase D: summarize filtered covers with cover-analyzer pieces.
        # n_tp_hits comes from our E-filtered (and possibly intersected) hit
        # sets, so --common is honored even though cover-analyzer.py would
        # otherwise count sibling *.out hits without a threshold.
        summary: dict[str, dict] = {}
        for t in targets:
            display = t["display"]
            rows = cover_analyzer.parse_file(filtered_covers[display])
            covered_sum = sum(r["covered"] for r in rows)
            true_sum = sum(r["true_length"] for r in rows)
            over_sum = sum(r["over"] for r in rows)
            percents = [r["percent"] for r in rows]
            over_percents = [r["over_percent"] for r in rows]
            n_rows = len(rows)
            if common is not None:
                n_tp = len(common)
                n_fp_decoy = len(common_decoy_fp)
                n_fp_nondecoy = len(common_nondecoy_fp)
                # Fragment counts are per-tool (common intersection has no
                # fragment identity); report per-tool raw counts separately.
                n_fp_background = per_tool_background_frags[display]
                n_fp_nondecoy_frags = per_tool_nondecoy_fp_frags[display]
            else:
                n_tp = len(per_tool_best[display])
                n_fp_decoy = len(per_tool_decoy_fp_best[display])
                n_fp_nondecoy = len(per_tool_nondecoy_fp_best[display])
                n_fp_background = per_tool_background_frags[display]
                n_fp_nondecoy_frags = per_tool_nondecoy_fp_frags[display]
            summary[display] = {
                "n_tp_hits": n_tp,
                "n_fp_decoy": n_fp_decoy,
                "n_fp_background": n_fp_background,
                "n_fp_nondecoy": n_fp_nondecoy,
                "n_fp_nondecoy_frags": n_fp_nondecoy_frags,
                "covered_sum": covered_sum,
                "true_length_sum": true_sum,
                "weighted_percent": (covered_sum / true_sum * 100
                                     if true_sum > 0 else 0),
                "mean_row_percent": (sum(percents) / n_rows
                                     if n_rows > 0 else 0),
                "weighted_over_percent": (over_sum / true_sum * 100
                                          if true_sum > 0 else 0),
                "mean_over_percent": (sum(over_percents) / n_rows
                                      if n_rows > 0 else 0),
                "mean_over": (over_sum / n_rows if n_rows > 0 else 0),
            }
        if args.latex:
            cover_analyzer.render_latex(summary, args.over_percent)
        else:
            cover_analyzer.render_text(summary, args.over_percent)
    finally:
        if args.keep_tmp:
            print(f"[coverage-filtered] kept staging dir {tmp_root}",
                  file=sys.stderr)
        else:
            shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
