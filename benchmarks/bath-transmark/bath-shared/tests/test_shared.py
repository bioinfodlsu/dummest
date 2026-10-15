#!/usr/bin/env python3
"""Tests for bath-shared hitcounts + style (stdlib unittest)."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hitcounts as hc  # noqa: E402
import style as st  # noqa: E402


class TestPairCounting(unittest.TestCase):
    def test_add_hit_keeps_min_e(self):
        hits, pb = [], {}
        hc.add_hit(hits, pb, "r1", "F", 1e-5, 10, 0)
        hc.add_hit(hits, pb, "r1", "F", 1e-7, 12, 0)
        self.assertEqual(pb, {("r1", "F"): 1e-7})
        self.assertEqual(len(hits), 2)

    def test_add_hit_none_read_no_pair(self):
        hits, pb = [], {}
        hc.add_hit(hits, pb, None, "F", 1e-5, 10, 0)
        self.assertEqual(hits, [(None, "F", 1e-5, 10, 0)])
        self.assertEqual(pb, {})

    def test_pair_best_from_hits_accept(self):
        hits = [("a", "F", 1e-3, 1, 0), ("b", "F", 1e-3, 1, 0)]
        pb = hc.pair_best_from_hits(hits, accept=lambda h: h[0] == "a")
        self.assertEqual(pb, {("a", "F"): 1e-3})

    def test_pairs_at_inclusive(self):
        pb = {("a", "F"): 1e-3, ("b", "F"): 2e-3}
        self.assertEqual(hc.pairs_at(pb, 1e-3), {("a", "F")})

    def test_at_threshold_inclusive(self):
        hits = [("a", "F", 1e-3, 1, 0)]
        self.assertEqual(
            hc.at_threshold(hits, {("a", "F"): 1e-3}, 1e-3), (1, 1))

    def test_unique_pairs_skips_none(self):
        hits = [("a", "F", 1, 1, 0), (None, "F", 1, 1, 0)]
        self.assertEqual(hc.unique_pairs(hits), {("a", "F")})
        self.assertEqual(hc.count_unique(hits), 1)


class TestPredicates(unittest.TestCase):
    def test_decoy(self):
        self.assertTrue(hc.is_decoy("decoy"))
        self.assertFalse(hc.is_decoy("Bac_GDH/71"))
        self.assertTrue(hc.is_non_decoy(None))

    def test_true_positive(self):
        self.assertTrue(hc.is_true_positive("Bac_GDH", "Bac_GDH/71", "same"))
        self.assertFalse(hc.is_true_positive("Bac_GDH", "Bac_GDH/71", "opposite"))
        self.assertFalse(hc.is_true_positive("Bac_GDH", "decoy", "same"))
        self.assertFalse(hc.is_true_positive("Bac_GDH", "Bac_GDH/x", "same"))


class TestTransmarkParser(unittest.TestCase):
    def _write(self, text):
        fd, path = tempfile.mkstemp(suffix=".out")
        with os.fdopen(fd, "w") as f:
            f.write(text)
        self.addCleanup(os.remove, path)
        return path

    def test_classified_9col(self):
        path = self._write(
            "0 10 1 2 read1 FamA 0 FamA/1 same\n"
            "0 9 3 4 read2 FamA 0 decoy decoy\n"
        )
        hits, pb, cls = hc.parse_transmark_out(path)
        self.assertEqual(len(hits), 2)
        self.assertEqual(cls, [("FamA/1", "same"), ("decoy", "decoy")])
        nd = hc.nondecoy_hits(hits, cls)
        self.assertEqual(len(nd), 1)
        self.assertEqual(pb, {("read1", "FamA"): 0.0, ("read2", "FamA"): 0.0})

    def test_unclassified_tmp_9col(self):
        path = self._write("1e-5 10 1 2 read1 FamA 0 0 fs\n")
        hits, _pb, cls = hc.parse_transmark_out(path)
        self.assertEqual(cls, [(None, None)])

    def test_classified_11col(self):
        path = self._write("1e-5 10 1 2 read1 FamA 0 0 fs FamA/2 same\n")
        _hits, _pb, cls = hc.parse_transmark_out(path)
        self.assertEqual(cls, [("FamA/2", "same")])

    def test_out_tmp_wrapper(self):
        path = self._write("1e-5 10 1 2 read1 FamA 0 0 fs\n")
        hits, pb = hc.parse_transmark_out_tmp(path)
        self.assertEqual(pb, {("read1", "FamA"): 1e-5})


class TestSweep(unittest.TestCase):
    def test_grid_and_tables(self):
        grid = hc.log_sweep_grid(1e-10, 1.0, 3)  # [1e-10, 1e-5, 1.0]
        self.assertEqual(len(grid), 3)
        counts = hc.sweep_table([("T", {("a", "F"): 1e-5})], grid)
        self.assertEqual(counts["T"], [0, 1, 1])
        bp = hc.bp_sweep_table([("T", [(1e-5, 10), (1e-3, 5)])], grid)
        self.assertEqual(bp["T"], [0, 10, 15])

    def test_parse_sweep_thresholds(self):
        self.assertEqual(hc.parse_sweep_thresholds("1e-3, 1e-10,1e-5"),
                         [1e-10, 1e-5, 1e-3])


class TestStyle(unittest.TestCase):
    def test_label_for_path(self):
        cases = {
            "dummer-fs-p002-analysis.xy": "DUMMEST (more sensitive)",
            "dummer-fs-analysis.xy": "DUMMEST",
            "bath-fs-no-fs-analysis.xy": "BATH --no-fs",
            "bath-fs-analysis.xy": "BATH",
            "x/hmmer_roc.xy": "HMMER3",
            "last-fs-analysis.xy": "LAST",
            "transmark-full-fs-results-last/last-fs-analysis.cover": "LAST",
            "unknown.xy": None,
        }
        for path, want in cases.items():
            self.assertEqual(st.label_for_path(path), want)

    def test_label_for_run(self):
        self.assertEqual(st.label_for_run("hmmsearch"), "HMMER3")
        self.assertEqual(st.label_for_run("bath"), "BATH")
        self.assertEqual(st.label_for_run("bath --nofs"), "BATH --no-fs")
        self.assertEqual(st.label_for_run("dummer"), "DUMMEST")
        self.assertEqual(st.label_for_run("last"), "LAST")
        self.assertEqual(st.label_for_run("lastal"), "LAST")

    def test_colors_and_aliases(self):
        self.assertEqual(st.color_for_tool("BATH --nofs"), "tab:orange")
        self.assertEqual(st.color_for_tool("BATH --no-fs"), "tab:orange")
        self.assertEqual(st.color_for_tool("dummer"), "tab:green")
        self.assertEqual(st.color_for_tool("DUMMEST"), "tab:green")
        self.assertEqual(st.color_for_tool("LAST"), "tab:brown")
        self.assertEqual(st.color_for_tool("last"), "tab:brown")
        self.assertEqual(st.canonical_tool("BATH --nofs"), "BATH --no-fs")
        self.assertIsNone(st.color_for_tool(None))
        # removed alias must not resolve
        self.assertEqual(st.canonical_tool("dum_seed"), "dum_seed")


if __name__ == "__main__":
    unittest.main()
