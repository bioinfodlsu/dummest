# bath-shared

Shared helpers for `BATH-paper/transmark` and `ncbi-data the seqkit is slow`,
added to both repos as a git submodule so there is no duplicated logic:

* `style.py` — canonical tool labels + `TOOL_COLORS` + `ALIASES`
  (`BATH --nofs` → `BATH --no-fs`) + `canonical_tool()` / `color_for_tool()`,
  plus `label_for_path()` (transmark filenames) and `label_for_run()` (ncbi
  runner/tool keys).
* `hitcounts.py` — the `(hits, pair_best)` contract and pair counting:
  `add_hit()` (single source of the min-E update),
  `pair_best_from_hits()` / `pairs_at()`,
  `parse_transmark_out()` (raw + classified) and `parse_transmark_out_tmp()`
  (unfiltered, full rank), `is_decoy()` / `is_true_positive()` predicates,
  `unique_pairs()` / `count_unique()`, `at_threshold()` (inclusive `E <= T`),
  and the E-value sweep helpers.

Every source parser funnels through `add_hit()`, so the tuple shape and the
min-E update live in exactly one place. Source adapters stay in each repo
(ncbi `common.py` for hmmer/bath/dummer, transmark `.out` here) and inject
source-specific behaviour via predicates (decoy filter is transmark-only).

Threshold convention is inclusive (`E <= T`) everywhere; this fixes the
old off-by-one where `compare_methods._fams_pairs_at_threshold` used
strict `<` while the sweep used `bisect_right` (`<=`).

Run the tests with:

    python3 -m unittest discover -s tests
