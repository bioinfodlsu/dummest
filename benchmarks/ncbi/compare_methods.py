#!/usr/bin/env python3
"""3-way comparison: hmmsearch vs BATH vs DUMMER no-seeds.

Reads existing output directories and reports per-family comparison at a
given E-value threshold, (read, family) pair-level agreement, and an
always-on E-value sweep (cumulative pairs and cumulative alignment bp).

Usage:
    python compare_methods.py <hmmer_dir> <bath_dir> <dummer_dir> [--evalue 0.001]
    Sweep: --sweep-max 10 --sweep-tsv sweep.tsv
           --sweep-plot sweep.png --bp-tsv bp.tsv
           [--bath-no-fs-dir bath_nofs]
"""

import argparse
import itertools
import os

import common
from bath_shared.hitcounts import (
    bp_sweep_table,
    log_sweep_grid,
    pairs_at,
    sweep_table,
    write_bp_tsv,
    write_sweep_tsv,
)
from bath_shared.style import color_for_tool, label_for_run


def _bump(fams, fam, evalue, score, span=0):
    if fam not in fams:
        fams[fam] = {'count': 0, 'best_e': evalue, 'best_s': score,
                     'total_len': 0}
    fams[fam]['count'] += 1
    fams[fam]['total_len'] += span
    if evalue < fams[fam]['best_e']:
        fams[fam]['best_e'] = evalue
    if score > fams[fam]['best_s']:
        fams[fam]['best_s'] = score


def _fams_pairs_at_threshold(hits, pair_best, threshold):
    """Build thresholded (fams, pairs) from unthresholded parse results.

    fams counts every passing hit; pairs contains each (read, family)
    once iff its smallest E-value passes (E <= threshold, INCLUSIVE).
    Hits with read None (DUMMER single-'s' blocks) contribute to fams only.
    """
    fams = {}
    for read, fam, evalue, score, span in hits:
        if evalue > threshold:
            continue
        _bump(fams, fam, evalue, score, span)
    return fams, pairs_at(pair_best, threshold)


def plot_sweep(path, tool_names, grid, counts, bp_counts=None):
    """Two stacked panels sharing the E-value x-axis: cumulative pairs
    on top, cumulative alignment bp below; one color per tool."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if bp_counts is None:
        fig, ax = plt.subplots()
        panels = [(ax, counts, 'Cumulative unique (read, family) pairs')]
    else:
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(6.4, 7.2),
                                             sharex=True)
        panels = [(ax_top, counts, 'Cumulative unique (read, family) pairs'),
                  (ax_bot, bp_counts, 'Cumulative alignment (bp)')]
    for ax, data, ylabel in panels:
        for name in tool_names:
            kw = {"marker": "o", "markersize": 3, "label": name}
            shared_color = color_for_tool(name)
            if shared_color is not None:
                kw["color"] = shared_color
            ax.semilogx(grid, data[name], **kw)
        ax.set_ylabel(ylabel)
        ax.set_xlim(min(grid), max(grid))
        ax.set_ylim(bottom=0)
        ax.grid(True, which='both', linestyle=':', alpha=0.6)
        ax.legend()
    panels[-1][0].set_xlabel('E-value threshold')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compare hmmsearch / BATH / DUMMER results at a given E-value threshold"
    )
    parser.add_argument("hmmer_dir", help="hmmsearch output directory")
    parser.add_argument("bath_dir", help="BATH output directory")
    parser.add_argument("dummer_dir", help="DUMMER (no-seeds) output directory")
    parser.add_argument("--evalue", type=float, default=0.001, help="E-value threshold (default: 0.001)")
    parser.add_argument("--sweep-max", type=float, default=10.0,
                        help="Sweep grid maximum E-value (default: 10)")
    parser.add_argument("--sweep-tsv", default=None,
                        help="Write sweep cumulative-count table to this TSV path")
    parser.add_argument("--sweep-plot", default=None,
                        help="Write sweep cumulative-curve PNG to this path "
                             "(requires matplotlib)")
    parser.add_argument("--bp-tsv", default=None,
                        help="Write cumulative-bp table to this TSV path")
    parser.add_argument("--bath-no-fs-dir", default=None,
                        help="BATH --no-fs output directory (bathsearch without "
                             "--fs); adds a 'BATH --no-fs' tool column")
    args = parser.parse_args()

    t = args.evalue

    hmm_hits, hmm_best = common.parse_hmmer_tbl(
        os.path.join(args.hmmer_dir, "hmmsearch_tbl.txt"))
    bath_hits, bath_best = common.parse_bath_tbl(
        os.path.join(args.bath_dir, "bath_tbl.txt"))
    vocab = ({fam for _, fam, _, _, _ in hmm_hits}
             | {fam for _, fam, _, _, _ in bath_hits})
    dn_hits, dn_best = common.parse_dummer(
        os.path.join(args.dummer_dir, "dummer_output.txt"), vocab)

    hmm, hmm_pairs = _fams_pairs_at_threshold(hmm_hits, hmm_best, t)
    bath, bath_pairs = _fams_pairs_at_threshold(bath_hits, bath_best, t)
    dn, dn_pairs = _fams_pairs_at_threshold(dn_hits, dn_best, t)

    # HMMER tblout carries no alignment length; fill family bp totals from
    # the domtblout (aa coords x3 = bp). Missing file -> stays 0.
    hmm_dom = common.parse_hmmer_dom(
        os.path.join(args.hmmer_dir, "hmmsearch_dom.txt"))
    if hmm_dom:
        for (read, fam), rows in hmm_dom.items():
            if fam in hmm:
                hmm[fam]['total_len'] += sum(
                    s for e, s in rows if e <= t)

    hmm_label = label_for_run("hmmsearch")
    bath_label = label_for_run("bath")
    tools = [(hmm_label, hmm, hmm_pairs), (bath_label, bath, bath_pairs)]
    sweep_bests = [(hmm_label, hmm_best), (bath_label, bath_best)]
    bp_inputs = [(hmm_label, [row for rows in hmm_dom.values()
                              for row in rows]),
                 (bath_label, [(e, s) for _, _, e, _, s in bath_hits])]
    if args.bath_no_fs_dir is not None:
        bn_label = label_for_run("bath --no-fs")
        bn_hits, bn_best = common.parse_bath_tbl(
            os.path.join(args.bath_no_fs_dir, "bath_tbl.txt"))
        bn, bn_pairs = _fams_pairs_at_threshold(bn_hits, bn_best, t)
        tools.append((bn_label, bn, bn_pairs))
        sweep_bests.append((bn_label, bn_best))
        bp_inputs.append(
            (bn_label, [(e, s) for _, _, e, _, s in bn_hits]))
    dn_label = label_for_run("dummer")
    tools.append((dn_label, dn, dn_pairs))
    sweep_bests.append((dn_label, dn_best))
    bp_inputs.append(
        (dn_label, [(e, s) for _, _, e, _, s in dn_hits]))
    all_fams = sorted(set().union(*(d for _, d, _ in tools)))

    # ── Summary ──
    print(f"{'Tool':>12s}  {'Families':>9s}  {'Hits':>8s}  "
          f"{'Pairs':>8s}  {'Total bp':>10s}")
    print("-" * 52)
    for n, d, p in tools:
        total_bp = sum(v.get('total_len', 0) for v in d.values())
        print(f"{n:>12s}  {len(d):>9d}  {sum(v['count'] for v in d.values()):>8d}"
              f"  {len(p):>8d}  {total_bp:>10d}")
    print()

    # ── Full table ──
    name_w = max(8, max(len(n) for n, _, _ in tools))
    hdr = f"{'Family':<55s}"
    for n, _, _ in tools:
        hdr += f"  {n:>{name_w}s}  len  E=best"
    print(hdr)
    print("-" * (55 + len(tools) * (name_w + 16)))
    for fam in all_fams:
        row = [fam[:54]]
        for _, d, _ in tools:
            e = d.get(fam, {}).get('best_e')
            if e is not None:
                c = d[fam]['count']
                l = d[fam].get('total_len', 0)
                if e < 1e-50:
                    row.append(f"{c:>4d} {l:>6d} {e:.0e}")
                elif e <= 0.001:
                    row.append(f"{c:>4d} {l:>6d} {e:.2e}")
                else:
                    row.append(f"{c:>4d} {l:>6d} {e:.2e}")
            else:
                row.append(f"  -      -      -")
        print("  ".join(row))

    # ── Overlap matrix (family-level, legacy) ──
    print()
    print(f"=== Family overlap (pairwise intersections, E<{t}) ===")
    fam_sets = [set(d.keys()) for _, d, _ in tools]
    for i, j in itertools.combinations(range(len(tools)), 2):
            inter = fam_sets[i] & fam_sets[j]
            only_i = fam_sets[i] - fam_sets[j]
            only_j = fam_sets[j] - fam_sets[i]
            print(f"  {tools[i][0]:>12s} ∩ {tools[j][0]:<12s} = {len(inter):>2d}"
                  f"  | {len(only_i):>2d} only in {tools[i][0]}"
                  f"  | {len(only_j):>2d} only in {tools[j][0]}")

    # ── Pair-level agreement: intersection of (read id, family) ──
    print()
    print(f"=== (read, family) pair agreement (E<{t}) ===")
    pair_sets = [p for _, _, p in tools]
    for i, j in itertools.combinations(range(len(tools)), 2):
            inter = pair_sets[i] & pair_sets[j]
            only_i = pair_sets[i] - pair_sets[j]
            only_j = pair_sets[j] - pair_sets[i]
            union = pair_sets[i] | pair_sets[j]
            jac = len(inter) / len(union) if union else 1.0
            print(f"  {tools[i][0]:>12s} ∩ {tools[j][0]:<12s} = {len(inter):>4d} pairs"
                  f"  (Jaccard {jac:.2f})"
                  f"  | {len(only_i):>4d} only in {tools[i][0]}"
                  f"  | {len(only_j):>4d} only in {tools[j][0]}")

    # ── Disagreement digest: families with the most tool-exclusive pairs ──
    print()
    print("=== Top families by (read, family) pair volume "
          "(per-tool pair counts) ===")
    fam_excl: dict[str, dict[str, int]] = {}
    for n, _, p in tools:
        for _, fam in p:
            fam_excl.setdefault(fam, {}).setdefault(n, 0)
            fam_excl[fam][n] += 1
    ranked = sorted(
        fam_excl.items(),
        key=lambda kv: (sum(kv[1].values()) - min(kv[1].values())
                        if len(kv[1]) > 1 else sum(kv[1].values())),
        reverse=True,
    )[:15]
    for fam, counts in ranked:
        parts = " / ".join(f"{tools[k][0]}:{counts.get(tools[k][0], 0)}"
                           for k in range(len(tools)))
        print(f"  {fam[:50]:50s}  {parts}")

    # ── Key clinical families ──
    print()
    print("=== Key clinical families detected ===")
    clinical = [
        'KPC', 'NDM', 'VIM', 'IMP', 'CTX-M', 'SHV', 'TEM',
        'MCR', 'OXA-48', 'OXA-23', 'OXA-24', 'OXA-10', 'OXA-1',
        'Erm', 'Cfr', 'MecA', 'vanA', 'vanB',
        'tetracycline', 'RND', 'ABC', 'SMR', 'MFS',
    ]
    for pat in clinical:
        found = []
        for n, d, _ in tools:
            matches = [f for f in d if pat.lower() in f.lower()]
            if matches:
                e_strs = [f"{d[m]['best_e']:.1e}" for m in matches]
                m_str = ", ".join(f"{m}({e_strs[i]})" for i, m in enumerate(matches))
                found.append(f"{n}({len(matches)}: {m_str})")
        if found:
            print(f"  {pat:>20s}: {' / '.join(found)}")

    # ── E-value sweep: cumulative pairs by smallest E per (read, family) ──
    # Each pair contributes once, at min(E) over its hits. Curve value at
    # threshold T = #{pairs with best_E <= T}. Always runs.
    print()
    print("=== E-value sweep: cumulative (read, family) pairs "
          "(smallest E per pair) ===")
    grid = log_sweep_grid(1e-50, args.sweep_max, 41)
    tool_names = [n for n, _ in sweep_bests]
    counts = sweep_table(sweep_bests, grid)
    print("threshold\t" + "\t".join(tool_names))
    for i, g in enumerate(grid):
        print(f"{g:.6e}\t" + "\t".join(str(counts[n][i])
                                        for n in tool_names))
    if args.sweep_tsv is not None:
        write_sweep_tsv(args.sweep_tsv, tool_names, grid, counts)
        print(f"\nSweep TSV written: {args.sweep_tsv}")

    # ── BP sweep: cumulative alignment bp over hits with E <= T ──
    # Curve value at threshold T = sum of spans of all hits with
    # E <= T (any such hit's pair trivially qualifies). Always runs.
    print()
    print("=== BP sweep: cumulative alignment bp (hit E <= T) ===")
    bp_names = [n for n, _ in bp_inputs]
    bp_counts = bp_sweep_table(bp_inputs, grid)
    print("threshold\t" + "\t".join(bp_names))
    for i, g in enumerate(grid):
        print(f"{g:.6e}\t" + "\t".join(str(bp_counts[n][i])
                                        for n in bp_names))
    if args.bp_tsv is not None:
        write_bp_tsv(args.bp_tsv, bp_names, grid, bp_counts)
        print(f"\nBP TSV written: {args.bp_tsv}")
    if args.sweep_plot is not None:
        try:
            plot_sweep(args.sweep_plot, tool_names, grid, counts,
                       bp_counts=bp_counts)
            print(f"Sweep plot written: {args.sweep_plot}")
        except ImportError as exc:
            print(f"WARNING: could not write sweep plot "
                  f"({exc}); TSVs above still valid.")


if __name__ == "__main__":
    main()
