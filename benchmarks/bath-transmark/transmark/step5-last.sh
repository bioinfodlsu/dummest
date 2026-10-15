#!/bin/bash
# Step 5: run the LAST benchmark with the restored x-lastal wrapper
# (original pre-dummer x-last from 26e573a; x-last itself now runs dummer).
# Usage: ./step5-last.sh prepare {full|lite|full-over|lite-over}
#          query DB: <qbase>.AA.fa/.last*
#        ./step5-last.sh run [--clean] {full|lite|full-over|lite-over}
#          train on reads, then lastal + pp. Default trains/benchmarks the
#          frameshifted Badread reads (<base>-fs.*); --clean uses the
#          pristine step-1 files (<base>.*) instead (matches steps 3/4).
# prepare depends only on the stable query MSA (run once); run retrains
# on the target reads each time. NOTE: $qbase.AA.train is query-keyed
# (x-lastal derives it from the query DB), so fs/clean runs on the same
# qbase overwrite it -- run sequentially, not in parallel.
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh

USAGE="Usage: $0 prepare {full|lite|full-over|lite-over} | $0 run [--clean] {full|lite|full-over|lite-over}"
[[ $# -ge 2 && $# -le 3 ]] || { echo "$USAGE" >&2; exit 1; }
MODE="$1"; shift
FS=1
if [[ "${1:-}" == "--clean" ]]; then FS=0; shift; fi
if [[ "${1:-}" == "--fs" ]]; then echo "step5-last: --fs is deprecated (fs is the default)" >&2; shift; fi
BASE="${1:-}"
case "$MODE" in
    prepare|run) ;;
    *) echo "$USAGE" >&2; exit 1 ;;
esac
case "$BASE" in
    full) qbase="$FULL_BASE" ;;
    lite) qbase="$LITE_BASE" ;;
    full-over) qbase="$FULL_OVER_BASE" ;;
    lite-over) qbase="$LITE_OVER_BASE" ;;
    *) echo "$USAGE" >&2; exit 1 ;;
esac
if [[ "$FS" -eq 1 ]]; then tbase="$qbase-fs"; else tbase="$qbase"; fi

LASTDB="$LAST_DIR/lastdb"
LASTTRAIN="$LAST_DIR/last-train"
ESLREFORMAT="$LAST_EASEL_DIR/miniapps/esl-reformat"
ESLSEQSTAT="$LAST_EASEL_DIR/miniapps/esl-seqstat"
for tool in "$LASTDB" "$LASTTRAIN" "$ESLREFORMAT" "$ESLSEQSTAT"; do
    [[ -x "$tool" ]] || { echo "step5-last: missing executable $tool -- check LAST_DIR/LAST_EASEL_DIR in config.sh" >&2; exit 1; }
done

case "$MODE" in
    prepare)
        [[ -f "$qbase.AA.msa" ]] || { echo "step5-last: missing $qbase.AA.msa -- run ./step2-prepare.sh first" >&2; exit 1; }
        "$ESLREFORMAT" fasta "$qbase.AA.msa" > "$qbase.AA.fa"
        echo "step5-last: wrote $qbase.AA.fa from $qbase.AA.msa"
        "$LASTDB" -P"$BENCH_CPUS" -q -c "$qbase.AA.last" "$qbase.AA.fa"
        echo "step5-last: built lastdb $qbase.AA.last* from $qbase.AA.fa"
        ;;
    run)
        for f in "$qbase.AA.last.prj" "$qbase.tbl" "$tbase.fa" "$tbase.pos"; do
            [[ -e "$f" ]] || { echo "step5-last: missing $f -- run ./step5-last.sh prepare $BASE first" >&2; exit 1; }
        done
        resultdir="./$tbase-results-last"
        # rocplot (via transmark-pp.sh) reads <tbase>.tbl + <tbase>.pos from one
        # basename. The .tbl is a step-1 query table, invariant across namespaces,
        # so link it (never copy) instead of duplicating query files.
        [[ -e "$tbase.tbl" ]] || ln -s "$qbase.tbl" "$tbase.tbl"
        mkdir -p "$resultdir"
        if [[ "$FS" -eq 1 ]]; then analysis="last-fs-analysis"; else analysis="last-analysis"; fi

        # Train the codon model on this run's target reads (x-lastal derives
        # the train path from the query DB and dies without it). Timed into
        # $tbase.train.time ("<tbase> <secs> seconds") so transmark-pp.sh
        # sums it with lastal's $tbase.time via transmark-time.pl.
        train_start=$SECONDS
        "$LASTTRAIN" -P"$BENCH_CPUS" --codon "$qbase.AA.last" "$tbase.fa" > "$qbase.AA.train"
        train_secs=$((SECONDS - train_start))
        printf '%s %d seconds\n' "$tbase" "$train_secs" > "$resultdir/$tbase.train.time"
        echo "step5-last: wrote $qbase.AA.train (trained on $tbase.fa) in ${train_secs}s"

        # x-lastal (10th arg) wants the average target length for E-value
        # scaling ($emax = $targetlen / $emax); derive it via esl-seqstat
        # (README convention) instead of asking the caller.
        targetlen="$("$ESLSEQSTAT" "$tbase.fa" | grep '^Average length:' | awk '{print $3}')"
        [[ -n "$targetlen" ]] || { echo "step5-last: could not parse average length from $tbase.fa via esl-seqstat" >&2; exit 1; }
        echo "step5-last: average target length for $tbase.fa: $targetlen"

        ./x-lastal "$LAST_DIR" . "$resultdir" "$tbase" \
            "$qbase.AA.last" "$tbase.pos" "$tbase.fa" "$BENCH_CPUS" "$LAST_EMAX" "$targetlen" \
        && ./transmark-pp.sh . "$tbase" "$resultdir" "$analysis" last
        ;;
esac
