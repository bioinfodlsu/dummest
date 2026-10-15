#!/usr/bin/env python3
"""
Fast, deterministic unit tests for the Transmark Badread helper scripts.

These exercise pos_bed_converter.py and the pure-Python logic inside
map_gene_reads.py (coordinate/CIGAR math and overlap computation). They do
NOT run the Badread simulator, so they finish in well under a second. The
end-to-end tests that do invoke Badread live in test_integration.py.

Run:
    python3 test_helpers.py
Exits non-zero if any test fails.
"""

import contextlib
import gzip
import io
import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HELPER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(HELPER_DIR))

from pos_bed_converter import pos_to_bed, bed_to_pos  # noqa: E402

try:
    from map_gene_reads import map_coordinate, get_overlaps, get_compression_type  # noqa: E402
    HAVE_MAPPER = True
    MAPPER_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on environment
    HAVE_MAPPER = False
    MAPPER_IMPORT_ERROR = exc


def quiet_stderr(func, *args, **kwargs):
    """Run func while suppressing its stderr warnings; return its result."""
    with contextlib.redirect_stderr(io.StringIO()):
        return func(*args, **kwargs)


@unittest.skipUnless(HAVE_MAPPER, f"map_gene_reads import failed: {MAPPER_IMPORT_ERROR}")
class MapCoordinateTests(unittest.TestCase):
    def test_pure_match(self):
        self.assertEqual(map_coordinate('8=', 0, 0), 0)
        self.assertEqual(map_coordinate('8=', 0, 7), 7)

    def test_mismatch_op(self):
        # 4=1X3= : q=4 is the first base of the X op
        self.assertEqual(map_coordinate('4=1X3=', 0, 4), 4)
        self.assertEqual(map_coordinate('4=1X3=', 0, 7), 7)

    def test_insertion_collapses(self):
        # 2=3I2= : query-only bases 2..4 all map to the same target position
        self.assertEqual(map_coordinate('2=3I2=', 0, 2), 2)
        self.assertEqual(map_coordinate('2=3I2=', 0, 3), 2)
        self.assertEqual(map_coordinate('2=3I2=', 0, 4), 2)
        # first base of the trailing 2= op
        self.assertEqual(map_coordinate('2=3I2=', 0, 5), 2)
        self.assertEqual(map_coordinate('2=3I2=', 0, 6), 3)

    def test_deletion_advances_target(self):
        # 2=3D2= : target skips 3 bases after the first match
        self.assertEqual(map_coordinate('2=3D2=', 0, 2), 5)
        self.assertEqual(map_coordinate('2=3D2=', 0, 3), 6)

    def test_start_offset(self):
        self.assertEqual(map_coordinate('3=', 10, 0), 10)
        self.assertEqual(map_coordinate('3=', 10, 2), 12)

    def test_last_base(self):
        self.assertEqual(map_coordinate('5=', 0, 4), 4)

    def test_past_end_returns_final_target_pos(self):
        self.assertEqual(map_coordinate('8=', 0, 8), 8)
        self.assertEqual(map_coordinate('2=3I2=', 0, 99), 4)


@unittest.skipUnless(HAVE_MAPPER, f"map_gene_reads import failed: {MAPPER_IMPORT_ERROR}")
class GetOverlapsTests(unittest.TestCase):
    N = 1000

    def test_interior(self):
        self.assertEqual(get_overlaps(100, 200, 50, 250, self.N), [(50, 150)])

    def test_partial_left(self):
        self.assertEqual(get_overlaps(100, 200, 0, 160, self.N), [(100, 160)])

    def test_no_overlap(self):
        self.assertEqual(get_overlaps(100, 200, 0, 50, self.N), [])

    def test_read_covers_gene(self):
        self.assertEqual(get_overlaps(100, 200, 0, 300, self.N), [(100, 200)])

    def test_read_end_at_contig_end_is_linear_path(self):
        # r_end == N must be treated as non-wrapping
        self.assertEqual(get_overlaps(900, 1000, 800, self.N, self.N), [(100, 200)])

    def test_circular_wraps_into_segment2(self):
        # gene [0,100) and read [950,1050) on a 1000 bp circle
        self.assertEqual(get_overlaps(0, 100, 950, 1050, self.N), [(50, 100)])

    def test_circular_wraps_into_segment1(self):
        # gene [950,1000) and read [900,1100)
        self.assertEqual(get_overlaps(950, 1000, 900, 1100, self.N), [(50, 100)])

    def test_circular_no_overlap(self):
        self.assertEqual(get_overlaps(500, 600, 900, 1100, self.N), [])


class CompressionTypeTests(unittest.TestCase):
    @unittest.skipUnless(HAVE_MAPPER, "map_gene_reads unavailable")
    def test_plain_and_gzip_and_empty(self):
        with tempfile.TemporaryDirectory() as td:
            plain = os.path.join(td, 'plain.fastq')
            with open(plain, 'w') as fh:
                fh.write('@r\nACGT\n+\nIIII\n')
            self.assertEqual(get_compression_type(plain), 'plain')

            gz = os.path.join(td, 'reads.fastq.gz')
            with gzip.open(gz, 'wt') as fh:
                fh.write('@r\nACGT\n+\nIIII\n')
            self.assertEqual(get_compression_type(gz), 'gz')

            empty = os.path.join(td, 'empty.fastq')
            open(empty, 'w').close()
            self.assertEqual(get_compression_type(empty), 'plain')


class PosBedConverterTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.td, ignore_errors=True)

    def _write(self, name, text):
        path = os.path.join(self.td, name)
        with open(path, 'w') as fh:
            fh.write(text)
        return path

    def _read(self, path):
        with open(path) as fh:
            return fh.read()

    def test_plus_roundtrip(self):
        pos = self._write('a.pos', 'FamA/1    read1    FamA/1    100    200\n')
        bed = os.path.join(self.td, 'a.bed')
        quiet_stderr(pos_to_bed, pos, bed)
        self.assertEqual(self._read(bed).split(), ['read1', '99', '200', 'FamA/1|FamA/1', '0', '+'])
        pos2 = os.path.join(self.td, 'a2.pos')
        quiet_stderr(bed_to_pos, bed, pos2)
        self.assertEqual(self._read(pos2).split(), ['FamA/1', 'read1', 'FamA/1', '100', '200'])

    def test_minus_roundtrip(self):
        pos = self._write('b.pos', 'FamA/2    read2    FamA/2    500    400\n')
        bed = os.path.join(self.td, 'b.bed')
        quiet_stderr(pos_to_bed, pos, bed)
        self.assertEqual(self._read(bed).split(), ['read2', '399', '500', 'FamA/2|FamA/2', '0', '-'])
        pos2 = os.path.join(self.td, 'b2.pos')
        quiet_stderr(bed_to_pos, bed, pos2)
        self.assertEqual(self._read(pos2).split(), ['FamA/2', 'read2', 'FamA/2', '500', '400'])

    def test_pipe_packed_name_roundtrip(self):
        pos = self._write('c.pos', 'A/B    read3    C/D    10    20\n')
        bed = os.path.join(self.td, 'c.bed')
        quiet_stderr(pos_to_bed, pos, bed)
        self.assertEqual(self._read(bed).split(), ['read3', '9', '20', 'A/B|C/D', '0', '+'])
        pos2 = os.path.join(self.td, 'c2.pos')
        quiet_stderr(bed_to_pos, bed, pos2)
        self.assertEqual(self._read(pos2).split(), ['A/B', 'read3', 'C/D', '10', '20'])

    def test_pos_skips_short_comment_and_bad_coords(self):
        pos = self._write('d.pos', (
            '# comment\n'
            '\n'
            'FamA/1 read1 FamA/1 10\n'          # too few columns
            'FamA/1 read1 FamA/1 x 20\n'        # non-integer coord
            'FamA/1 read1 FamA/1 10 20\n'       # valid
        ))
        bed = os.path.join(self.td, 'd.bed')
        quiet_stderr(pos_to_bed, pos, bed)
        lines = [ln for ln in self._read(bed).splitlines() if ln.strip()]
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith('read1\t9\t20\t'))

    def test_bed_skips_non_bed6_and_unknown_strand(self):
        bed = self._write('e.bed', (
            'track name=x\n'
            'browser position chr1\n'
            '# comment\n'
            '\n'
            'read1\t9\t20\tA|B\t0\n'          # BED5, not enough columns
            'read2\t9\t20\tA|B\t0\t.\n'       # unknown strand -> skipped
            'read3\t9\t20\tA|B\t0\t+\n'       # valid
        ))
        pos = os.path.join(self.td, 'e.pos')
        quiet_stderr(bed_to_pos, bed, pos)
        lines = [ln for ln in self._read(pos).splitlines() if ln.strip()]
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0].split(), ['A', 'read3', 'B', '10', '20'])

    def test_extra_columns_ignored(self):
        pos = self._write('f.pos', 'FamA/1 read1 FamA/1 10 20 extra1 extra2\n')
        bed = os.path.join(self.td, 'f.bed')
        quiet_stderr(pos_to_bed, pos, bed)
        self.assertEqual(self._read(bed).split(), ['read1', '9', '20', 'FamA/1|FamA/1', '0', '+'])


@unittest.skipUnless(HAVE_MAPPER, f"map_gene_reads import failed: {MAPPER_IMPORT_ERROR}")
class MapperClampRegressionTests(unittest.TestCase):
    """Regression for the 3' +1 overshoot: when a gene's end coincides with the
    fragment end and that terminal base is a read-internal insertion (present in
    the reference fragment, absent from the read), the mapper used to emit
    read_end == len(read) + 1. Coordinates must stay within [0, len(read)]."""

    MAPPER = str(HELPER_DIR / 'map_gene_reads.py')

    def _run(self, ref, header, readseq, gene_line, tmp_dir):
        with open(os.path.join(tmp_dir, 'ref.fa'), 'w') as fh:
            fh.write('>seq1\n' + ref + '\n')
        with open(os.path.join(tmp_dir, 'reads.fastq'), 'w') as fh:
            fh.write(f'@r1 {header}\n{readseq}\n+\n' + 'I' * len(readseq) + '\n')
        with open(os.path.join(tmp_dir, 'genes.bed'), 'w') as fh:
            fh.write(gene_line + '\n')
        out = os.path.join(tmp_dir, 'out.tsv')
        subprocess.run([sys.executable, self.MAPPER,
                        '--reference', os.path.join(tmp_dir, 'ref.fa'),
                        '--reads', os.path.join(tmp_dir, 'reads.fastq'),
                        '--genes', os.path.join(tmp_dir, 'genes.bed'),
                        '--output', out],
                       stderr=subprocess.DEVNULL, check=True)
        with open(out) as fh:
            rows = [ln.split('\t') for ln in fh.read().splitlines()[1:]]
        return rows

    def setUp(self):
        random.seed(1)
        self.ref = ''.join(random.choice('ACGT') for _ in range(200))
        self.rc = self.ref.translate(str.maketrans('ACGT', 'TGCA'))[::-1]
        self.td = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.td, ignore_errors=True)

    def test_plus_trailing_insertion(self):
        rows = self._run(self.ref, 'seq1,+strand,0-50 length=49', self.ref[0:49],
                         'seq1\t10\t50\tgeneX\t0\t+', self.td)
        self.assertEqual(len(rows), 1)
        start, end = int(rows[0][2]), int(rows[0][3])
        self.assertEqual((start, end), (10, 49))
        self.assertLessEqual(end, 49)

    def test_minus_trailing_insertion(self):
        rows = self._run(self.ref, 'seq1,-strand,0-50 length=49', self.rc[0:49],
                         'seq1\t150\t190\tgeneY\t0\t-', self.td)
        self.assertEqual(len(rows), 1)
        start, end = int(rows[0][2]), int(rows[0][3])
        self.assertLessEqual(end, 49)
        self.assertLess(start, end)

    def test_normal_no_overshoot(self):
        rows = self._run(self.ref, 'seq1,+strand,0-50 length=50', self.ref[0:50],
                         'seq1\t10\t50\tgeneX\t0\t+', self.td)
        self.assertEqual(len(rows), 1)
        start, end = int(rows[0][2]), int(rows[0][3])
        self.assertEqual((start, end), (10, 50))


@unittest.skipUnless(HAVE_MAPPER, f"map_gene_reads import failed: {MAPPER_IMPORT_ERROR}")
class MapperGuardAndFilterTests(unittest.TestCase):
    """Guard against non-aligning reads and the --min-length sliver filter."""

    MAPPER = str(HELPER_DIR / 'map_gene_reads.py')

    def setUp(self):
        self.td = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.td, ignore_errors=True)

    def _run(self, ref, header, readseq, gene_line, extra=None):
        with open(os.path.join(self.td, 'ref.fa'), 'w') as fh:
            fh.write('>seq1\n' + ref + '\n')
        with open(os.path.join(self.td, 'reads.fastq'), 'w') as fh:
            fh.write(f'@r1 {header}\n{readseq}\n+\n' + 'I' * len(readseq) + '\n')
        with open(os.path.join(self.td, 'genes.bed'), 'w') as fh:
            fh.write(gene_line + '\n')
        out = os.path.join(self.td, 'out.tsv')
        cmd = [sys.executable, self.MAPPER,
               '--reference', os.path.join(self.td, 'ref.fa'),
               '--reads', os.path.join(self.td, 'reads.fastq'),
               '--genes', os.path.join(self.td, 'genes.bed'),
               '--output', out] + (extra or [])
        subprocess.run(cmd, stderr=subprocess.DEVNULL, check=True)
        with open(out) as fh:
            return [ln.split('\t') for ln in fh.read().splitlines()[1:]]

    def test_non_aligning_read_is_skipped(self):
        # read shares no bases with the fragment -> all-insertion alignment
        rows = self._run('C' * 200, 'seq1,+strand,0-50 length=50', 'A' * 50,
                         'seq1\t10\t50\tgeneX\t0\t+')
        self.assertEqual(rows, [])

    def test_min_length_default_drops_sliver(self):
        # gene [10,50) only overlaps the fragment [45,95) by 5 bp
        ref = ''.join(random.Random(7).choice('ACGT') for _ in range(200))
        rows = self._run(ref, 'seq1,+strand,45-95 length=50', ref[45:95],
                         'seq1\t10\t50\tgeneX\t0\t+')
        self.assertEqual(rows, [])

    def test_min_length_override_keeps_sliver(self):
        ref = ''.join(random.Random(7).choice('ACGT') for _ in range(200))
        rows = self._run(ref, 'seq1,+strand,45-95 length=50', ref[45:95],
                         'seq1\t10\t50\tgeneX\t0\t+', extra=['--min-length', '1'])
        self.assertEqual(len(rows), 1)
        self.assertEqual((int(rows[0][2]), int(rows[0][3])), (0, 5))

    def test_min_length_keeps_normal_overlap(self):
        ref = ''.join(random.Random(7).choice('ACGT') for _ in range(200))
        # gene [10,60) overlaps fragment [45,95) by 15 bp
        rows = self._run(ref, 'seq1,+strand,45-95 length=50', ref[45:95],
                         'seq1\t10\t60\tgeneX\t0\t+')
        self.assertEqual(len(rows), 1)
        self.assertEqual(int(rows[0][3]) - int(rows[0][2]), 15)


if __name__ == '__main__':
    if not HAVE_MAPPER:
        print(f"WARNING: map_gene_reads could not be imported ({MAPPER_IMPORT_ERROR}); "
              f"mapper tests will be skipped. Install the 'badread' Python package.",
              file=sys.stderr)
    unittest.main(verbosity=2)
