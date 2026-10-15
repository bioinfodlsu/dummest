#!/bin/bash
# Step 4: run the dummer benchmark (replaces test4.sh).
# Usage: ./step4-dummer.sh [--gapped-pvalue <x>] [--ungapped-pvalue <x>] [--max] [--clean] [full|lite|full-over|lite-over|all]
# Default benchmarks the frameshifted Badread reads (<base>-fs.*, from
# step2-frameshift.sh + step2-prepare.sh).
# --clean benchmarks the pristine step-1 files (<base>.*) instead.
# --max/--gapped-pvalue/--ungapped-pvalue are passed to dummest;
# outputs get a suffix (-max, -p01, ...) so variants never overwrite each other.
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh

EXTRA_FLAGS=""
FS=1
SUFFIX=""
VARIANT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean) FS=0; shift ;;
        --fs) echo "step4-dummer: --fs is deprecated (fs is now the default; use --clean for pristine data)" >&2; FS=1; shift ;;
        --max) EXTRA_FLAGS="$EXTRA_FLAGS --max"; SUFFIX="${SUFFIX}-max"; shift ;;
        --gapped-pvalue)
            if [[ $# -lt 2 || ! "$2" =~ ^[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$ ]]; then
                echo "step4-dummer: --gapped-pvalue needs a numeric value" >&2; exit 1; fi
            EXTRA_FLAGS="$EXTRA_FLAGS --gapped-pvalue $2"
            SUFFIX="$SUFFIX-p${2//./}"
            shift 2 ;;
        --ungapped-pvalue)
            if [[ $# -lt 2 || ! "$2" =~ ^[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$ ]]; then
                echo "step4-dummer: --ungapped-pvalue needs a numeric value" >&2; exit 1; fi
            EXTRA_FLAGS="$EXTRA_FLAGS --ungapped-pvalue $2"
            SUFFIX="$SUFFIX-u${2//./}"
            shift 2 ;;
        full|lite|full-over|lite-over|all) VARIANT="$1"; shift ;;
        *) break ;;
    esac
done

fsbase() { if [[ "$FS" -eq 1 ]]; then echo "$1-fs"; else echo "$1"; fi; }

run_one() {
    local qbase="$1" tbase="$2"
    local resultdir="./$tbase-results-v2$SUFFIX" analysis="dummer"
    if [[ "$FS" -eq 1 ]]; then analysis="$analysis-fs"; fi
    analysis="$analysis$SUFFIX-analysis"
    for f in "$qbase.AA.hmm" "$qbase.AA.msa" "$qbase.tbl" "$tbase.fa" "$tbase.pos"; do
        [[ -f "$f" ]] || { echo "step4-dummer: missing $f -- run ./step1-create.sh, ./step2-frameshift.sh and ./step2-prepare.sh as needed" >&2; exit 1; }
    done
    # rocplot (via transmark-pp.sh) reads <tbase>.tbl + <tbase>.pos from one
    # basename. The .tbl is a step-1 query table, invariant across namespaces,
    # so link it (never copy) instead of duplicating query files.
    [[ -e "$tbase.tbl" ]] || ln -s "$qbase.tbl" "$tbase.tbl"
    mkdir -p "$resultdir"
    # Auto-repair: drop stale .fai (Badread UUIDs change per frameshift run;
    # a committed index has 0 overlap and makes bedtools skip all windows).
    # bedtools getfasta recreates it on demand.
    rm -f "$tbase.fa.fai" "$tbase.fa.fai.old"
    # Reuse step3-dummer-precompute.sh outputs for full-fs / full-over-fs
    # when present: <tbase>.6frame.pfa + <tbase>.targetDB_pad* ->
    # dummest --target-db-pad. Lite (-over) targets stay per-run.
    PREBUILT_ARGS=""
    if [[ "$tbase" == "$FULL_BASE-fs" || "$tbase" == "$FULL_OVER_BASE-fs" ]]; then
        if [[ -f "$tbase.6frame.pfa" && -f "$tbase.targetDB_pad" && -f "$tbase.targetDB_pad.index" && -f "$tbase.targetDB_pad.dbtype" ]]; then
            # Inline asserts + freshness (auto-repair): prebuilt must carry
            # frame= headers and must not be older than the reads.
            if ! head -1 "$tbase.6frame.pfa" | grep -q 'frame='; then
                echo "step4-dummer: stale prebuilt $tbase.6frame.pfa (missing frame=, seqkit -F dropped?) -- falling back to per-run tmpdir; run ./step3-dummer-precompute.sh --force" >&2
            elif [[ "$tbase.fa" -nt "$tbase.6frame.pfa" || "$tbase.6frame.pfa" -nt "$tbase.targetDB_pad" ]]; then
                echo "step4-dummer: stale prebuilt DB for $tbase ($tbase.fa newer than $tbase.6frame.pfa/targetDB_pad) -- falling back to per-run tmpdir; run ./step3-dummer-precompute.sh --force" >&2
            else
                echo "step4-dummer: using precomputed 6-frame fa ($tbase.6frame.pfa) + target DB ($tbase.targetDB_pad) for $tbase"
                PREBUILT_ARGS="--target-db-pad $tbase.targetDB_pad"
            fi
        else
            echo "step4-dummer: no precomputed target DB for $tbase (missing $tbase.6frame.pfa / $tbase.targetDB_pad*), pipeline will build per-run in tmpdir" >&2
        fi
    fi
    # shellcheck disable=SC2086
    ./x-last "$DUMMER_DIR" . "$resultdir" "$tbase" \
        "$qbase.AA.hmm" "$tbase.pos" "$tbase.fa" "$BENCH_CPUS" "$DUMMER_EMAX" $EXTRA_FLAGS $PREBUILT_ARGS \
    && ./transmark-pp.sh . "$tbase" "$resultdir" "$analysis" dummer
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
    *) echo "Usage: $0 [--gapped-pvalue <x>] [--ungapped-pvalue <x>] [--max] [--clean] [full|lite|full-over|lite-over|all]" >&2; exit 1 ;;
esac
