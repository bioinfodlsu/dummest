#!/usr/bin/env python3
"""
Audit a map_gene_reads.py output (mapped.tsv) against its reads and gene BED.

Checks that every mapped interval is self-consistent and within bounds:
  * read_id exists in the reads file
  * 0 <= read_start < read_end <= len(read)   (catches the 3' +1 overshoot)
  * read_strand is '+' or '-'
  * the (ref_contig, ref_start, ref_end, gene_name) matches a BED gene
  * read_gene_strand is the genomic strand, flipped on -strand reads
  * no duplicate (read, gene, start, end, strand) rows

Exits non-zero if any violation is found, so it can gate a pipeline or a test.

Usage:
    python3 audit_mapping.py --mapped mapped.tsv --reads reads.fastq --genes genes.bed
    python3 audit_mapping.py --dir tmp-frameshift/transmark-full   # infers the trio

The reads file may be FASTQ or FASTA, plain or gzipped.
"""

import argparse
import gzip
import sys
from collections import Counter


def open_maybe_gzip(path):
    with open(path, 'rb') as fh:
        magic = fh.read(2)
    if magic == b'\x1f\x8b':
        return gzip.open(path, 'rt')
    return open(path, 'r')


def load_reads(path):
    """Return {read_name: (length, header_meta_or_None)} for FASTQ/FASTA (plain or gz).

    header_meta is the second whitespace field of the record header (Badread's
    "contig,strand,start-end"), or None for FASTA / unrecognised headers.
    """
    reads = {}
    opener = gzip.open if path.endswith('.gz') else open
    try:
        with opener(path, 'rt') as fh:
            first = fh.readline()
            if not first:
                return reads
            if first.startswith('>'):
                name, seq_len = None, 0
                for line in [first] + fh.readlines():
                    if line.startswith('>'):
                        if name is not None:
                            reads[name] = (seq_len, None)
                        name = line[1:].split()[0]
                        seq_len = 0
                    else:
                        seq_len += len(line.strip())
                if name is not None:
                    reads[name] = (seq_len, None)
            elif first.startswith('@'):
                fh.seek(0)
                while True:
                    head = fh.readline()
                    if not head:
                        break
                    seq = fh.readline()
                    fh.readline()
                    fh.readline()
                    if not head.startswith('@'):
                        continue
                    parts = head.strip()[1:].split()
                    meta = parts[1] if len(parts) > 1 else None
                    reads[parts[0]] = (len(seq.strip()), meta)
            else:
                raise ValueError(f'unrecognised reads format (first line: {first[:20]!r})')
    except UnicodeDecodeError as exc:
        raise ValueError(f'could not read {path} as text (gzipped? use .gz name)') from exc
    return reads


def load_genes(path):
    """Return {contig: [(start, end, name, strand)]}."""
    genes = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('track') or line.startswith('browser'):
                continue
            parts = line.split('\t')
            if len(parts) < 4:
                parts = line.split()
                if len(parts) < 4:
                    continue
            contig = parts[0]
            start, end = int(parts[1]), int(parts[2])
            name = parts[3]
            strand = parts[5] if len(parts) >= 6 else '+'
            genes.setdefault(contig, []).append((start, end, name, strand))
    return genes


def audit(mapped_path, reads_path, genes_path):
    reads = load_reads(reads_path)
    genes = load_genes(genes_path)
    gene_set = {f'{c}|{s}|{e}|{n}' for c, vs in genes.items() for s, e, n, _ in vs}
    gene_strand = {}
    for c, vs in genes.items():
        for s, e, n, st in vs:
            gene_strand[(c, s, e, n)] = st

    counters = Counter()
    samples = {}

    def bump(key, detail):
        counters[key] += 1
        samples.setdefault(key, detail)

    with open(mapped_path) as fh:
        header = fh.readline().rstrip('\n').split('\t')
        expected = ['read_id', 'gene_name', 'read_start', 'read_end',
                    'read_gene_strand', 'ref_contig', 'ref_start', 'ref_end']
        if header != expected:
            print(f'WARNING: unexpected header: {header}', file=sys.stderr)
        for ln in fh:
            if not ln.strip():
                continue
            f = ln.rstrip('\n').split('\t')
            if len(f) != 8:
                bump('malformed_row', ln.strip())
                continue
            counters['rows'] += 1
            rid, gname = f[0], f[1]
            rs, re_ = int(f[2]), int(f[3])
            gene_read_strand = f[4]
            contig = f[5]
            gstart, gend = int(f[6]), int(f[7])

            if rid not in reads:
                bump('missing_read_id', rid)
                continue
            n, meta = reads[rid]

            if rs < 0:
                bump('negative_start', f'{rid} {rs}')
            if rs >= re_:
                bump('start_ge_end', f'{rid} {rs}..{re_}')
            if re_ > n:
                key = 'overshoot_by_1' if re_ == n + 1 else 'overshoot'
                bump(key, f'{rid} read_end={re_} len={n}')
            if not (0 <= rs < re_ <= n):
                bump('out_of_range', f'{rid} {rs}..{re_} len={n}')

            if gene_read_strand not in ('+', '-'):
                bump('bad_gene_strand', f'{rid} {gene_read_strand}')

            if f'{contig}|{gstart}|{gend}|{gname}' not in gene_set:
                bump('ref_not_in_genes', f'{rid} {contig} {gstart}-{gend} {gname}')

            # read_gene_strand must be the genomic gene strand, flipped on -strand reads
            gs = gene_strand.get((contig, gstart, gend, gname))
            read_own_strand = None
            if meta:
                mp = meta.split(',')
                if len(mp) >= 2 and mp[1] in ('+strand', '-strand'):
                    read_own_strand = '+' if mp[1] == '+strand' else '-'
            if gs is not None and read_own_strand is not None:
                expect = gs if read_own_strand == '+' else ('-' if gs == '+' else '+')
                if gene_read_strand != expect:
                    bump('strand_mismatch',
                         f'{rid} {gname} read={read_own_strand} got={gene_read_strand} expect={expect}')

            counters['_rowsig'] += 1
            counters['_dupkey_' + '|'.join(f)] += 1

    dup = sum(1 for k, v in counters.items() if k.startswith('_dupkey_') and v > 1)

    failures = {k: v for k, v in counters.items()
                if not k.startswith('_') and k not in ('rows',)}
    total = sum(failures.values())

    print(f'audit_mapping: {mapped_path}')
    print(f'  reads:  {reads_path} ({len(reads)} sequences)')
    print(f'  genes:  {genes_path}')
    print(f'  rows:   {counters["rows"]}')
    print(f'  duplicate rows: {dup}')
    if failures:
        print('  VIOLATIONS:')
        for k in sorted(failures):
            print(f'    {k}: {failures[k]}  (e.g. {samples.get(k)})')
    else:
        print('  no violations')
    return total + dup


def main():
    ap = argparse.ArgumentParser(description='Audit a map_gene_reads.py mapped.tsv.')
    ap.add_argument('--mapped', help='mapped.tsv from map_gene_reads.py')
    ap.add_argument('--reads', help='reads FASTQ/FASTA (plain or .gz)')
    ap.add_argument('--genes', help='gene BED file')
    ap.add_argument('--dir', help='step2 scratch dir; infers mapped.tsv/reads.fastq/genes.bed')
    args = ap.parse_args()

    mapped = args.mapped
    reads = args.reads
    genes = args.genes
    if args.dir:
        mapped = mapped or f'{args.dir}/mapped.tsv'
        reads = reads or f'{args.dir}/reads.fastq'
        genes = genes or f'{args.dir}/genes.bed'
    if not (mapped and reads and genes):
        ap.error('provide --mapped/--reads/--genes or --dir')

    problems = audit(mapped, reads, genes)
    sys.exit(1 if problems else 0)


if __name__ == '__main__':
    main()
