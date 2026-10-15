#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

VARIANT="${1:-full}"
source ./config.sh

gen_one() {
    local tag="$1"; shift
    local out="modern_rocplot-${tag}-fs.png"
    for f in "$@"; do
        [[ -f "$f" ]] || { echo "step-last-generate-figures: missing $f -- run ./step3-bathsearch.sh, ./step4-dummer.sh and ./step5-last.sh first" >&2; exit 1; }
    done
    python3 plot.py "$@"
    mv -f modern_rocplot.png "$out"
    echo "step-last-generate-figures: wrote $out from: $*"
}

gen_cover_one() {
    local tag="$1"; shift
    local txt="coverage-${tag}-fs.txt"
    local tex="coverage-${tag}-fs.tex"
    for f in "$@"; do
        [[ -f "$f" ]] || { echo "step-last-generate-figures: missing $f -- run ./step3-bathsearch.sh, ./step4-dummer.sh and ./step5-last.sh first" >&2; exit 1; }
    done
    python3 cover-analyzer.py "$@" > "$txt"
    python3 cover-analyzer.py --latex "$@" > "$tex"
    echo "step-last-generate-figures: wrote $txt and $tex from: $*"
}

gen_evalue_one() {
    local tag="$1" bath_out="$2" dummer_out="$3"
    local out="evalue-scatter-${tag}-fs.png"
    for f in "$bath_out" "$dummer_out"; do
        [[ -f "$f" ]] || { echo "step-last-generate-figures: missing $f -- run ./step3-bathsearch.sh and ./step4-dummer.sh first" >&2; exit 1; }
    done
    python3 evalue-scatter.py "$bath_out" "$dummer_out" -o "$out"
    echo "step-last-generate-figures: wrote $out (BATH vs DUMMEST, ${tag})"
}

gen_full() {
    gen_one full \
        "transmark-full-fs-results/bath-fs-analysis.xy" \
        "transmark-full-fs-results-no-fs/bath-fs-no-fs-analysis.xy" \
        "transmark-full-fs-results-v2/dummer-fs-analysis.xy" \
        "transmark-full-fs-results-v2-p002/dummer-fs-p002-analysis.xy" \
        #"transmark-full-fs-results-last/last-fs-analysis.xy"
    gen_cover_one full \
        "transmark-full-fs-results/bath-fs-analysis.cover" \
        "transmark-full-fs-results-no-fs/bath-fs-no-fs-analysis.cover" \
        "transmark-full-fs-results-v2/dummer-fs-analysis.cover" \
        "transmark-full-fs-results-v2-p002/dummer-fs-p002-analysis.cover" \
        #"transmark-full-fs-results-last/last-fs-analysis.cover"
    gen_evalue_one full \
        "transmark-full-fs-results/transmark-full-fs.out" \
        "transmark-full-fs-results-v2/transmark-full-fs.out"
}

gen_over_one() {
    # $1 = tag (lite-over|full-over), $2 = qbase without -fs
    local tag="$1" qbase="$2"
    local tbase="$qbase-fs"
    gen_one "$tag" \
        "$tbase-results/bath-fs-analysis.xy" \
        "$tbase-results-no-fs/bath-fs-no-fs-analysis.xy" \
        "$tbase-results-v2/dummer-fs-analysis.xy" \
        #"$tbase-results-last/last-fs-analysis.xy"
    gen_cover_one "$tag" \
        "$tbase-results/bath-fs-analysis.cover" \
        "$tbase-results-no-fs/bath-fs-no-fs-analysis.cover" \
        "$tbase-results-v2/dummer-fs-analysis.cover" \
        #"$tbase-results-last/last-fs-analysis.cover"
    gen_evalue_one "$tag" \
        "$tbase-results/$tbase.out" \
        "$tbase-results-v2/$tbase.out"
}

case "$VARIANT" in
    full) gen_full ;;
    lite-over) gen_over_one "lite-over" "$LITE_OVER_BASE" ;;
    full-over) gen_over_one "full-over" "$FULL_OVER_BASE" ;;
    *) echo "Usage: $0 [full|lite-over|full-over]" >&2; exit 1 ;;
esac
