#!/usr/bin/env python3
"""Shared hit-counting helpers for BATH-paper/transmark and ncbi-data.

Common definitions (no duplicated logic across repos):
    raw    = number of parsed hit lines (full rank, unfiltered)
    unique = number of distinct (sequence-id, profile/family-id) pairs

    A pair counts once at its smallest E-value. Threshold comparisons
    are INCLUSIVE (E <= T) everywhere -- this fixes the historical
    off-by-one where compare_methods._fams_pairs_at_threshold used
    strict ``<`` while sweep_table (bisect_right) used ``<=``.

Every source parser funnels through :func:`add_hit` so the ``(hits,
pair_best)`` contract and the min-E update live in exactly one place.

Transmark adapters emit the same ``(read, family, evalue, score, span)``
tuple shape as ncbi-data common.py parsers, so threshold/sweep helpers
work on both. Transmark ``*.out.tmp`` columns (9, whitespace-delimited):
    0=evalue 1=bitscore 2=target_from 3=target_to 4=target(seq-id)
    5=query(profile/family) 6=fs 7=stop 8=pipe
``*.out`` appends ``[-2]=matching_hit [-1]=strand``; pair identity uses
only cols 4/5, so both files yield identical pair sets, but only ``.out``
carries the ground-truth classification needed for decoy filtering.
"""
from __future__ import annotations

import bisect
import glob
import itertools
import math
import os


def add_hit(hits, pair_best, read, family, evalue, score, span):
    """Record one hit in the shared ``(hits, pair_best)`` form.

    ``read is None`` (DUMMER single-'s' blocks) contributes to ``hits``
    (family counting) but never to ``pair_best``.
    """
    hits.append((read, family, evalue, score, span))
    if read is None:
        return
    key = (read, family)
    prev = pair_best.get(key)
    if prev is None or evalue < prev:
        pair_best[key] = evalue


def pair_best_from_hits(hits, accept=None):
    """Build ``{(read, family): min E}`` from a hit stream.

    ``accept`` is an optional predicate over the 5-tuple; hits for which
    it returns false are ignored. ncbi passes ``None``; transmark passes
    a non-decoy predicate (or pre-filters the list).
    """
    pair_best = {}
    for hit in hits:
        if accept is not None and not accept(hit):
            continue
        read, family, evalue = hit[0], hit[1], hit[2]
        if read is None:
            continue
        key = (read, family)
        prev = pair_best.get(key)
        if prev is None or evalue < prev:
            pair_best[key] = evalue
    return pair_best


def unique_pairs(hits):
    """Return the set of distinct (sequence-id, profile-id) pairs."""
    return {(read, fam) for read, fam, _e, _s, _span in hits if read is not None}


def count_unique(hits) -> int:
    """Return the number of distinct (sequence-id, profile-id) pairs."""
    return len(unique_pairs(hits))


def pairs_at(pair_best, threshold: float):
    """Return the set of pairs whose smallest E-value is <= threshold."""
    return {p for p, e in pair_best.items() if e <= threshold}


def at_threshold(hits, pair_best, threshold: float):
    """Count (n_raw, n_unique) with INCLUSIVE E-value cutoff (E <= T).

    n_raw    = hits with evalue <= T (None-read hits included).
    n_unique = pairs whose smallest E-value <= T.
    """
    n_raw = sum(1 for _read, _fam, e, _s, _span in hits if e <= threshold)
    return n_raw, len(pairs_at(pair_best, threshold))


# ---------------------------------------------------------------------------
# Transmark .out / .out.tmp source adapter
# ---------------------------------------------------------------------------

def parse_transmark_out(path: str):
    """Parse a transmark output file (``*.out`` classified / ``*.out.tmp`` raw).

    Returns ``(hits, pair_best, classified)``:
        hits       [(read, family, evalue, score, span)]
        pair_best  {(read, family): smallest_E}
        classified list parallel to ``hits`` of ``(matching_hit, strand)``
                   (both None for unclassified input)

    Classification is detected from the last field (``same``/``opposite``/
    ``decoy``), not the column count: ``.out`` is 9 or 11 columns depending
    on the tool, while ``.out.tmp`` is 7 or 9 and ends in fs/stop/pipe.

    Skips blank lines, ``#`` comments and short lines only. No filtering
    here; callers apply decoy/true-positive predicates afterwards.
    """
    strand_tokens = {"same", "opposite", "decoy"}
    hits = []
    pair_best = {}
    classified = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                evalue = float(parts[0])
            except ValueError:
                continue
            try:
                score = float(parts[1])
            except ValueError:
                score = float("-inf")
            read = parts[4]
            family = parts[5]
            add_hit(hits, pair_best, read, family, evalue, score, 0)
            if parts[-1] in strand_tokens:
                classified.append((parts[-2], parts[-1]))
            else:
                classified.append((None, None))
    return hits, pair_best, classified


def parse_transmark_out_tmp(path: str):
    """Parse a transmark ``*.out.tmp``/``*.out`` file, UNFILTERED.

    Backwards-compatible 2-tuple wrapper around :func:`parse_transmark_out`.
    """
    hits, pair_best, _classified = parse_transmark_out(path)
    return hits, pair_best


def is_decoy(matching_hit):
    """True when a transmark classification field is a decoy hit."""
    return matching_hit is not None and matching_hit.startswith("decoy")


def is_non_decoy(matching_hit):
    """True when a transmark hit is not a decoy (unclassified -> True)."""
    return not is_decoy(matching_hit)


def is_true_positive(family, matching_hit, strand):
    """Transmark true-positive rule (transmark-coverage.pl / -mer.pl).

    A hit is a true positive iff ``matching_hit`` is ``<family>/<digits>``
    for the query family and the strand is "same".
    """
    if matching_hit is None or strand != "same":
        return False
    prefix = family + "/"
    if not matching_hit.startswith(prefix):
        return False
    return matching_hit[len(prefix):].isdigit()


def nondecoy_hits(hits, classified):
    """Filter hits to those without a decoy classification."""
    return [h for h, (mh, _st) in zip(hits, classified) if not is_decoy(mh)]


# ---------------------------------------------------------------------------
# E-value sweep helpers
# ---------------------------------------------------------------------------

def log_sweep_grid(lo=1e-50, hi=10.0, n=41):
    """Log-spaced E-value grid from lo to hi."""
    if lo <= 0 or hi <= 0 or n < 2:
        raise ValueError("sweep grid needs lo>0, hi>0, n>=2")
    llo, lhi = math.log10(lo), math.log10(hi)
    return [10.0 ** (llo + i * (lhi - llo) / (n - 1)) for i in range(n)]


def parse_sweep_thresholds(spec):
    """Parse a comma-separated threshold spec into sorted floats."""
    vals = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok:
            vals.append(float(tok))
    if not vals:
        raise ValueError("empty sweep threshold spec")
    return sorted(vals)


def cumulative_counts(sorted_evals, grid):
    """For each threshold in grid, count evals with best_E <= threshold."""
    return [bisect.bisect_right(sorted_evals, g) for g in grid]


def sweep_table(pair_bests, grid):
    """Return {tool_name: [unique-pair counts per grid]} (E <= T)."""
    out = {}
    for name, pb in pair_bests:
        out[name] = cumulative_counts(sorted(pb.values()), grid)
    return out


def bp_sweep_table(bp_inputs, grid):
    """Return {tool_name: [summed bp of hits with E <= T per grid]}."""
    out = {}
    for name, rows in bp_inputs:
        ordered = sorted(rows, key=lambda r: r[0])
        evals = [e for e, _ in ordered]
        cumsum = list(itertools.accumulate(s for _, s in ordered))
        col = []
        for g in grid:
            k = bisect.bisect_right(evals, g)
            col.append(cumsum[k - 1] if k else 0)
        out[name] = col
    return out


def _write_counts_tsv(path, tool_names, grid, counts):
    with open(path, "w") as f:
        f.write("threshold\t" + "\t".join(tool_names) + "\n")
        for i, g in enumerate(grid):
            f.write(f"{g:.6e}\t" + "\t".join(
                str(counts[n][i]) for n in tool_names) + "\n")


def write_sweep_tsv(path, tool_names, grid, counts):
    _write_counts_tsv(path, tool_names, grid, counts)


def write_bp_tsv(path, tool_names, grid, counts):
    _write_counts_tsv(path, tool_names, grid, counts)


def find_transmark_hits(cover_path: str):
    """Return the sibling classified ``*.out`` (else ``*.out.tmp``) or None.

    ``.out`` is preferred because it carries the matching_hit/strand
    classification used for non-decoy filtering; both yield identical
    pair sets.
    """
    directory = os.path.dirname(os.path.abspath(cover_path))
    for pattern in ("*.out", "*.out.tmp"):
        matches = sorted(
            m for m in glob.glob(os.path.join(directory, pattern))
            if not m.endswith(".sort")
        )
        if matches:
            return matches[0]
    return None
