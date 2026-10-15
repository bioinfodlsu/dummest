#!/usr/bin/env python3
"""
Manual visual check of gene-boundary mapping on Badread reads.

Generates a synthetic reference with a marker-flanked gene, simulates reads
with Badread, finds one read whose parent fragment fully spans the gene, and
prints single-base alignments around the gene start and end. The mapped
boundary positions are bracketed; a human (or the printed MATCH/MISMATCH
summary) can confirm the reference and read characters agree.

This is a human-inspection aid, not an automated regression test - see
test_integration.py for assertions. Requires an upstream Badread clone
(BADREAD_DIR) and the badread + edlib Python packages.

Usage:
    python3 verify_boundaries.py                       # + strand gene, + strand read
    python3 verify_boundaries.py --strand -            # - strand gene, - strand read
    python3 verify_boundaries.py --seed 7 --ref-len 20000
"""

import argparse
import os
import random
import subprocess
import sys
from pathlib import Path

HELPER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(HELPER_DIR))

import edlib  # noqa: E402
from badread.misc import load_fasta, reverse_complement  # noqa: E402
from map_gene_reads import map_coordinate  # noqa: E402

BADREAD_DIR = os.environ.get('BADREAD_DIR', '/home/harvsftw/Documents/bioinfo/Badread')
BADREAD_RUNNER = os.path.join(BADREAD_DIR, 'badread-runner.py')


def random_dna(n, rng):
    return ''.join(rng.choice('ACGT') for _ in range(n))


def build_reference(args, rng):
    core = 'GCA' * args.core_repeats
    gene_seq = 'ATG' + args.marker_start + core + args.marker_end + 'TAA'
    left = random_dna(args.gene_start, rng)
    right = random_dna(args.ref_len - args.gene_start - len(gene_seq), rng)
    full = left + gene_seq + right
    gene_end = args.gene_start + len(gene_seq)

    with open(args.ref, 'w') as fh:
        fh.write(f'>{args.contig}\n')
        for i in range(0, len(full), 80):
            fh.write(full[i:i + 80] + '\n')
    with open(args.genes, 'w') as fh:
        fh.write(f'{args.contig}\t{args.gene_start}\t{gene_end}\tgmarker\t0\t{args.strand}\n')
    return gene_seq, gene_end


def get_aligned_pairs(query, target, cigar, start_loc):
    import re
    q_pos, t_pos, pairs = 0, start_loc, []
    for part in re.findall(r'\d+[\w=]', cigar):
        op_len, op_type = int(part[:-1]), part[-1]
        for _ in range(op_len):
            if op_type in ('=', 'X'):
                pairs.append((q_pos, t_pos, query[q_pos], target[t_pos], op_type))
                q_pos += 1
                t_pos += 1
            elif op_type == 'I':
                pairs.append((q_pos, None, query[q_pos], '-', 'I'))
                q_pos += 1
            elif op_type == 'D':
                pairs.append((None, t_pos, '-', target[t_pos], 'D'))
                t_pos += 1
    return pairs


def print_window(pairs, q_lo, q_hi, read_boundary):
    ref_line = align_line = read_line = ''
    for q_idx, t_idx, q_char, t_char, op in pairs:
        if q_idx is None or not (q_lo <= q_idx <= q_hi):
            continue
        is_boundary = (t_idx is not None and t_idx == read_boundary)
        ref_line += f'[{q_char}]' if is_boundary else f' {q_char} '
        read_line += f'[{t_char}]' if is_boundary else f' {t_char} '
        if is_boundary:
            align_line += '[|]' if op == '=' else '[.]'
        else:
            align_line += ' | ' if op == '=' else ' . '
    print('Ref:  ' + ref_line)
    print('      ' + align_line)
    print('Read: ' + read_line)
    return ref_line, read_line


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ref', default='_vb_ref.fasta')
    ap.add_argument('--genes', default='_vb_genes.bed')
    ap.add_argument('--reads', default='_vb_reads.fastq')
    ap.add_argument('--contig', default='seq1')
    ap.add_argument('--ref-len', type=int, default=10000)
    ap.add_argument('--gene-start', type=int, default=3000)
    ap.add_argument('--core-repeats', type=int, default=100)
    ap.add_argument('--marker-start', default='CCCCGGGG')
    ap.add_argument('--marker-end', default='AAAATTTT')
    ap.add_argument('--strand', choices=['+', '-'], default='+')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--quantity', default='100k')
    ap.add_argument('--length', default='4000,1000')
    ap.add_argument('--identity', default='90,95,3')
    ap.add_argument('--keep', action='store_true', help='keep temp files')
    args = ap.parse_args()

    if not os.path.isfile(BADREAD_RUNNER):
        sys.exit(f'verify_boundaries: badread-runner.py not found: {BADREAD_RUNNER} '
                 f'(set BADREAD_DIR)')

    rng = random.Random(args.seed)
    gene_seq, gene_end = build_reference(args, rng)
    requested = '+strand' if args.strand == '+' else '-strand'

    print(f'=== simulating reads (Badread, {args.quantity}, seed {args.seed}) ===')
    with open(args.reads, 'w') as fh:
        subprocess.run([sys.executable, BADREAD_RUNNER, 'simulate',
                        '--reference', args.ref, '--quantity', args.quantity,
                        '--length', args.length, '--identity', args.identity,
                        '--chimeras', '0', '--junk_reads', '0', '--random_reads', '0',
                        '--seed', str(args.seed)], stdout=fh, check=True)

    ref_seqs, _, _, _, _ = load_fasta(args.ref)
    refseq = ref_seqs[args.contig]
    rc = reverse_complement(refseq)
    n = len(refseq)

    found = False
    with open(args.reads) as fh:
        while True:
            head = fh.readline()
            if not head:
                break
            seq = fh.readline().strip()
            fh.readline()
            fh.readline()
            if not head.startswith('@'):
                continue
            f = head.strip()[1:].split()
            if len(f) < 2:
                continue
            contig, strand, coords = f[1].split(',')
            if contig != args.contig or strand != requested or '-' not in coords or '(' in coords:
                continue
            r_start, r_end = map(int, coords.split('-'))
            # need a read whose parent fragment fully contains the gene with margin
            if args.strand == '+':
                frag = refseq[r_start:r_end] if r_end <= n else refseq[r_start:] + refseq[:r_end - n]
                gs, ge = args.gene_start, gene_end
            else:
                frag = rc[r_start:r_end] if r_end <= n else rc[r_start:] + rc[:r_end - n]
                gs, ge = n - gene_end, n - args.gene_start
            if gs < r_start or ge > r_end:
                continue
            g_s_on_frag = gs - r_start
            g_e_on_frag = ge - r_start
            if not (0 <= g_s_on_frag < g_e_on_frag <= len(frag)):
                continue

            print(f'\nFound {requested} read spanning the gene: {head.strip()}')
            print(f'  chromosome fragment {r_start}..{r_end}')
            res = edlib.align(frag, seq, mode='HW', task='path')
            cigar, (start_loc, _) = res['cigar'], res['locations'][0]
            pairs = get_aligned_pairs(frag, seq, cigar, start_loc)
            read_gene_start = map_coordinate(cigar, start_loc, g_s_on_frag)
            read_gene_end = map_coordinate(cigar, start_loc, g_e_on_frag - 1) + 1

            print(f'\n--- gene START (fragment index {g_s_on_frag}) ---')
            print_window(pairs, g_s_on_frag - 8, g_s_on_frag + 8, read_gene_start)
            print(f'\n--- gene END (fragment index {g_e_on_frag - 1}) ---')
            print_window(pairs, g_e_on_frag - 10, g_e_on_frag + 6, read_gene_end - 1)

            # explicit character check via the alignment pairs
            def base_at(q_idx):
                for q, t, qc, tc, op in pairs:
                    if q == q_idx:
                        return tc, t
                return None, None
            s_read, s_t = base_at(g_s_on_frag)
            e_read, e_t = base_at(g_e_on_frag - 1)
            s_ref = frag[g_s_on_frag]
            e_ref = frag[g_e_on_frag - 1]
            print(f'\nstart base: ref={s_ref} read={s_read} '
                  f'({"MATCH" if s_read == s_ref else "mismatch/indel"})')
            print(f'end base:   ref={e_ref} read={e_read} '
                  f'({"MATCH" if e_read == e_ref else "mismatch/indel"})')
            found = True
            break
    if not found:
        print('No spanning read found; increase --quantity or lower --identity.')

    if not args.keep:
        for p in (args.ref, args.genes, args.reads):
            if os.path.exists(p):
                os.remove(p)


if __name__ == '__main__':
    main()
