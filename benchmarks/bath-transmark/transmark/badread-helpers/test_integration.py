#!/usr/bin/env python3
"""
End-to-end tests for map_gene_reads.py against real Badread output.

These build small synthetic references (linear, multi-gene, both strands, and
a circular contig), run `badread-runner.py simulate`, run the mapper, and check
that every emitted interval is (a) within the read bounds and (b) aligns to the
correct reference subsequence. They also check the chimera/junk/random skip
paths and gzip input handling.

These are slower than test_helpers.py because they invoke Badread. They are
skipped automatically if an upstream Badread clone (BADREAD_DIR) or the edlib
package is unavailable.

Run:
    BADREAD_DIR=/path/to/Badread python3 test_integration.py
    python3 test_integration.py -k linear        # single test
"""

import gzip
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HELPER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(HELPER_DIR))

try:
    import edlib
    from badread.misc import reverse_complement
    from map_gene_reads import get_overlaps
    HAVE_DEPS = True
    DEP_ERROR = None
except Exception as exc:  # pragma: no cover - depends on environment
    HAVE_DEPS = False
    DEP_ERROR = exc

BADREAD_DIR = os.environ.get('BADREAD_DIR', '/home/harvsftw/Documents/bioinfo/Badread')
BADREAD_RUNNER = os.path.join(BADREAD_DIR, 'badread-runner.py')
HAVE_BADREAD = os.path.isfile(BADREAD_RUNNER)
MAPPER = str(HELPER_DIR / 'map_gene_reads.py')

SKIP_REASON = None
if not HAVE_DEPS:
    SKIP_REASON = f'python deps unavailable: {DEP_ERROR}'
elif not HAVE_BADREAD:
    SKIP_REASON = f'badread-runner.py not found (set BADREAD_DIR): {BADREAD_RUNNER}'


def random_dna(n, rng):
    return ''.join(rng.choice('ACGT') for _ in range(n))


def write_fasta(path, records):
    """records: list of (name, sequence, [extra_header])"""
    with open(path, 'w') as fh:
        for name, seq, *extra in records:
            head = name + ((' ' + extra[0]) if extra else '')
            fh.write(f'>{head}\n')
            for i in range(0, len(seq), 80):
                fh.write(seq[i:i + 80] + '\n')


def parse_fastq(path):
    """Yield (read_id, header_field2, sequence) for each record."""
    with open(path) as fh:
        while True:
            head = fh.readline()
            if not head:
                break
            seq = fh.readline().strip()
            fh.readline()
            fh.readline()
            if not head.startswith('@'):
                continue
            parts = head.strip()[1:].split(' ')
            yield parts[0], (parts[1] if len(parts) > 1 else ''), seq


def fragment_for(refseq, rc_refseq, strand, r_start, r_end, n):
    """Reference chunk covering [r_start, r_end) in the read's orientation."""
    if strand == '+strand':
        if r_end <= n:
            return refseq[r_start:r_end]
        return refseq[r_start:] + refseq[:r_end - n]
    if r_end <= n:
        return rc_refseq[r_start:r_end]
    return rc_refseq[r_start:] + rc_refseq[:r_end - n]


@unittest.skipUnless(HAVE_DEPS and HAVE_BADREAD, SKIP_REASON)
class IntegrationBase(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.rng = random.Random(42)

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def path(self, name):
        return os.path.join(self.td, name)

    def run_badread(self, ref_path, out_path, length, identity='95,99,3', seed=42,
                    quantity='200k', chimeras='0', junk='0', rand='0'):
        cmd = [sys.executable, BADREAD_RUNNER, 'simulate',
               '--reference', ref_path,
               '--quantity', quantity,
               '--length', length,
               '--identity', identity,
               '--chimeras', chimeras,
               '--junk_reads', junk,
               '--random_reads', rand,
               '--seed', str(seed)]
        with open(out_path, 'w') as fh:
            subprocess.run(cmd, stdout=fh, check=True)

    def run_mapper(self, ref_path, reads_path, genes_path, out_path):
        subprocess.run([sys.executable, MAPPER,
                        '--reference', ref_path,
                        '--reads', reads_path,
                        '--genes', genes_path,
                        '--output', out_path],
                       stderr=subprocess.DEVNULL, check=True)

    def read_mapped(self, out_path):
        with open(out_path) as fh:
            header = fh.readline().rstrip('\n').split('\t')
            return header, [ln.rstrip('\n').split('\t') for ln in fh]

    def verify_rows(self, rows, ref_seqs, rc_seqs, genes, min_identity=70.0):
        """Check every mapped row against the reference and the read bases."""
        self.assertGreater(len(rows), 0, 'expected at least one mapped row')
        gene_by_contig = {}
        for chrom, start, end, name, strand in genes:
            gene_by_contig.setdefault(chrom, []).append((start, end, name, strand))

        for row in rows:
            rid, gname, rs, re_, rstrand, contig, gstart, gend = row
            rs, re_ = int(rs), int(re_)
            gstart, gend = int(gstart), int(gend)
            # interval must be valid on the read
            self.assertGreaterEqual(rs, 0, f'{rid}: negative read_start')
            self.assertGreater(re_, rs, f'{rid}: read_end <= read_start')
            self.assertIn(rstrand, ('+', '-'), f'{rid}: bad strand')
            # ref interval must match a gene on that contig
            self.assertIn((gstart, gend, gname), [(s, e, n) for s, e, n, _ in gene_by_contig[contig]])

        # re-derive expected overlaps from read headers and check sequence identity
        for row in rows:
            rid, gname, rs, re_, rstrand, contig, gstart, gend = row
            rs, re_ = int(rs), int(re_)
            header_meta = self.headers[rid]
            hc, hstrand, coords = header_meta.split(',')
            r_start, r_end = map(int, coords.split('-'))
            n = len(ref_seqs[hc])
            frag = fragment_for(ref_seqs[hc], rc_seqs[hc], hstrand, r_start, r_end, n)

            # gene coords on the read's orientation
            gS, gE = int(gstart), int(gend)
            if hstrand == '+strand':
                gs, ge = gS, gE
            else:
                gs, ge = n - gE, n - gS
            overlaps = get_overlaps(gs, ge, r_start, r_end, n)
            subseqs = [frag[o_s:o_e] for o_s, o_e in overlaps]

            seg = self.seqs[rid][rs:re_]
            best = max(
                (100.0 * (1.0 - edlib.align(sub, seg, mode='NW')['editDistance'] / max(len(sub), 1))
                 for sub in subseqs if sub),
                default=0.0)
            self.assertGreaterEqual(
                best, min_identity,
                f'{rid} {gname}: mapped [{rs},{re_}) does not align to the expected '
                f'reference subsequence (best identity {best:.1f}%)')


@unittest.skipUnless(HAVE_DEPS and HAVE_BADREAD, SKIP_REASON)
class LinearMultiGeneTests(IntegrationBase):
    def test_linear_plus_and_minus_genes(self):
        rng = self.rng
        n = 12000
        plus = random_dna(n, rng)
        # geneP on + strand, geneN on - strand
        gp = (2000, 3800, 'geneP', '+')
        gn = (7000, 8800, 'geneN', '-')
        ref_path = self.path('ref.fa')
        write_fasta(ref_path, [('seqA', plus)])
        genes_path = self.path('genes.bed')
        with open(genes_path, 'w') as fh:
            for s, e, nm, st in (gp, gn):
                fh.write(f'seqA\t{s}\t{e}\t{nm}\t0\t{st}\n')

        reads = self.path('reads.fastq')
        self.run_badread(ref_path, reads, length='5000,1500', quantity='400k', seed=11)

        ref_seqs = {'seqA': plus}
        rc_seqs = {'seqA': reverse_complement(plus)}
        out = self.path('mapped.tsv')
        self.run_mapper(ref_path, reads, genes_path, out)
        _, rows = self.read_mapped(out)

        self.headers, self.seqs = {}, {}
        for rid, meta, seq in parse_fastq(reads):
            self.headers[rid] = meta
            self.seqs[rid] = seq

        self.verify_rows(rows, ref_seqs, rc_seqs, [('seqA', *gp), ('seqA', *gn)])

        # read_gene_strand must be the genomic strand flipped on -strand reads
        gene_strand = {('geneP'): '+', ('geneN'): '-'}
        saw_plus_gene = saw_minus_gene = False
        for row in rows:
            rid, gname, rs, re_, rstrand, contig, gstart, gend = row
            expected = gene_strand[gname]
            if rstrand == '-':
                expected = '-' if expected == '+' else '+'
            self.assertEqual(row[4], expected, f'{rid} {gname}: wrong read_gene_strand')
            if gname == 'geneP':
                saw_plus_gene = True
            else:
                saw_minus_gene = True
        self.assertTrue(saw_plus_gene and saw_minus_gene, 'expected both genes to be covered')


@unittest.skipUnless(HAVE_DEPS and HAVE_BADREAD, SKIP_REASON)
class CircularWrapTests(IntegrationBase):
    def test_circular_reads_wrap_and_stay_in_range(self):
        rng = self.rng
        n = 2000
        seq = random_dna(n, rng)
        gene = (1200, 1900, 'geneC', '+')
        ref_path = self.path('ref.fa')
        write_fasta(ref_path, [('seqC', seq, 'circular=true')])
        genes_path = self.path('genes.bed')
        with open(genes_path, 'w') as fh:
            fh.write(f'seqC\t{gene[0]}\t{gene[1]}\t{gene[2]}\t0\t{gene[3]}\n')

        reads = self.path('reads.fastq')
        # reads near the contig length so many wrap around the origin
        self.run_badread(ref_path, reads, length='1800,150', quantity='120k', seed=5)

        out = self.path('mapped.tsv')
        self.run_mapper(ref_path, reads, genes_path, out)
        _, rows = self.read_mapped(out)

        self.headers, self.seqs = {}, {}
        for rid, meta, rseq in parse_fastq(reads):
            self.headers[rid] = meta
            self.seqs[rid] = rseq

        for row in rows:
            self.assertLessEqual(int(row[3]), len(self.seqs[row[0]]), 'coordinate past read end')
        self.verify_rows(rows, {'seqC': seq}, {'seqC': reverse_complement(seq)},
                         [('seqC', *gene)])


@unittest.skipUnless(HAVE_DEPS and HAVE_BADREAD, SKIP_REASON)
class SkipPathTests(IntegrationBase):
    def test_chimera_junk_random_are_skipped(self):
        rng = self.rng
        n = 6000
        seq = random_dna(n, rng)
        gene = (1500, 3500, 'geneD', '+')
        ref_path = self.path('ref.fa')
        write_fasta(ref_path, [('seqD', seq)])
        genes_path = self.path('genes.bed')
        with open(genes_path, 'w') as fh:
            fh.write(f'seqD\t{gene[0]}\t{gene[1]}\t{gene[2]}\t0\t{gene[3]}\n')

        reads = self.path('reads.fastq')
        self.run_badread(ref_path, reads, length='3000,500', quantity='200k', seed=7,
                         chimeras='10', junk='10', rand='10')

        out = self.path('mapped.tsv')
        self.run_mapper(ref_path, reads, genes_path, out)
        _, rows = self.read_mapped(out)

        self.headers = {}
        for rid, meta, seq_ in parse_fastq(reads):
            self.headers[rid] = meta
        for row in rows:
            meta = self.headers[row[0]]
            self.assertNotIn('chimera', meta)
            self.assertNotIn('junk_seq', meta)
            self.assertNotIn('random_seq', meta)

        self.seqs = {}
        for rid, meta, seq_ in parse_fastq(reads):
            self.seqs[rid] = seq_
        self.verify_rows(rows, {'seqD': seq}, {'seqD': reverse_complement(seq)},
                         [('seqD', *gene)])


class GzipEquivalenceTests(unittest.TestCase):
    """Mapper must produce identical output for plain and gzipped FASTQ.
    Built synthetically so it needs neither Badread nor network."""

    @unittest.skipUnless(HAVE_DEPS, SKIP_REASON)
    def test_gzip_matches_plain(self):
        rng = random.Random(3)
        n = 2000
        seq = random_dna(n, rng)
        rc = reverse_complement(seq)
        with tempfile.TemporaryDirectory() as td:
            ref_path = os.path.join(td, 'ref.fa')
            write_fasta(ref_path, [('seqG', seq)])
            genes_path = os.path.join(td, 'genes.bed')
            with open(genes_path, 'w') as fh:
                fh.write('seqG\t500\t1200\tgeneG\t0\t+\n')
            # a couple of exact reads (plain) via headers + sequences
            fastq_lines = []
            for i, (st, en, strand) in enumerate([(400, 1300, '+strand'), (900, 1700, '-strand')]):
                frag = fragment_for(seq, rc, strand, st, en, n)
                fastq_lines.append(f'@r{i} seqG,{strand},{st}-{en} length={len(frag)}')
                fastq_lines.append(frag)
                fastq_lines.append('+')
                fastq_lines.append('I' * len(frag))
            plain = os.path.join(td, 'reads.fastq')
            with open(plain, 'w') as fh:
                fh.write('\n'.join(fastq_lines) + '\n')
            gz = os.path.join(td, 'reads.fastq.gz')
            with gzip.open(gz, 'wt') as fh:
                fh.write('\n'.join(fastq_lines) + '\n')

            out_plain = os.path.join(td, 'plain.tsv')
            out_gz = os.path.join(td, 'gz.tsv')
            for reads_path, out_path in ((plain, out_plain), (gz, out_gz)):
                subprocess.run([sys.executable, MAPPER, '--reference', ref_path,
                                '--reads', reads_path, '--genes', genes_path,
                                '--output', out_path],
                               stderr=subprocess.DEVNULL, check=True)
            self.assertEqual(Path(out_plain).read_text(), Path(out_gz).read_text())


if __name__ == '__main__':
    if SKIP_REASON:
        print(f'WARNING: integration tests will be skipped: {SKIP_REASON}', file=sys.stderr)
    unittest.main(verbosity=2)
