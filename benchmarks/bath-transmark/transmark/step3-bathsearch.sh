#!/bin/bash
# Step 3: run the bathsearch benchmark (replaces test3.sh / test3-nofs.sh).
# Usage: ./step3-bathsearch.sh [--no-fs] [--max] [--clean] [full|lite|full-over|lite-over|all]   (default variant: all)
# Default benchmarks the frameshifted Badread reads (<base>-fs.*, from
# step2-frameshift.sh + step2-prepare.sh) with the frameshift-aware HMM.
# --no-fs runs the non-frameshift HMM on the same reads (control).
# --max disables all bathsearch heuristic filters (slow; lite only).
# --clean benchmarks the pristine step-1 files (<base>.*) instead.
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh

NOFS=0
MAX=0
FS=1
EXTRA_FLAGS=""
SUFFIX=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-fs) NOFS=1; shift ;;
        --nofs) echo "step3-bathsearch: --nofs is deprecated, use --no-fs" >&2; NOFS=1; shift ;;
        --max) MAX=1; EXTRA_FLAGS="$EXTRA_FLAGS --max"; SUFFIX="${SUFFIX}-max"; shift ;;
        --clean) FS=0; shift ;;
        --fs) echo "step3-bathsearch: --fs is deprecated (fs is now the default; use --clean for pristine data)" >&2; FS=1; shift ;;
        *) break ;;
    esac
done
VARIANT="${1:-all}"

if [[ "$MAX" -eq 1 && "$VARIANT" != "lite" && "$VARIANT" != "lite-over" ]]; then
    echo "step3-bathsearch: warning: --max on $VARIANT is very slow (all filters off)" >&2
fi

fsbase() { if [[ "$FS" -eq 1 ]]; then echo "$1-fs"; else echo "$1"; fi; }

run_one() {
    local qbase="$1" tbase="$2" mod="" abase xbath="./x-bathsearch"
    if [[ "$FS" -eq 1 ]]; then abase="bath-fs"; else abase="bath"; fi
    if [[ "$NOFS" -eq 1 ]]; then mod="-no-fs"; xbath="./x-bathsearch-no-fs"; fi
    local resultdir="$tbase-results$mod$SUFFIX" analysis="${abase}${mod}${SUFFIX}-analysis"
    for f in "$qbase.AA.bhmm" "$qbase.tbl" "$tbase.fa" "$tbase.pos"; do
        [[ -f "$f" ]] || { echo "step3-bathsearch: missing $f -- run ./step1-create.sh, ./step2-frameshift.sh and ./step2-prepare.sh as needed" >&2; exit 1; }
    done
    # rocplot (via transmark-pp.sh) reads <tbase>.tbl + <tbase>.pos from one
    # basename. The .tbl is a step-1 query table, invariant across namespaces,
    # so link it (never copy) instead of duplicating query files.
    [[ -e "$tbase.tbl" ]] || ln -s "$qbase.tbl" "$tbase.tbl"
    mkdir -p "$resultdir"
    # shellcheck disable=SC2086
    "$xbath" "$BATHSEARCH_DIR" . "./$resultdir" "$tbase" \
        "$qbase.AA.bhmm" "$tbase.pos" "$tbase.fa" "$BENCH_CPUS" "$BATH_EMAX" $EXTRA_FLAGS
    ./transmark-pp.sh . "$tbase" "./$resultdir" "$analysis" bathsearch
}

case "$VARIANT" in
    full) run_one "$FULL_BASE" "$(fsbase "$FULL_BASE")" ;;
    lite) run_one "$LITE_BASE" "$(fsbase "$LITE_BASE")" ;;
    full-over) run_one "$FULL_OVER_BASE" "$(fsbase "$FULL_OVER_BASE")" ;;
    lite-over) run_one "$LITE_OVER_BASE" "$(fsbase "$LITE_OVER_BASE")" ;;
    all)  run_one "$FULL_BASE" "$(fsbase "$FULL_BASE")"
          run_one "$LITE_BASE" "$(fsbase "$LITE_BASE")"
          run_one "$FULL_OVER_BASE" "$(fsbase "$FULL_OVER_BASE")"
          run_one "$LITE_OVER_BASE" "$(fsbase "$LITE_OVER_BASE")" ;;
    *) echo "Usage: $0 [--no-fs] [--max] [--clean] [full|lite|full-over|lite-over|all]" >&2; exit 1 ;;
esac
