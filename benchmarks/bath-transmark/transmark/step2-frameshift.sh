#!/bin/bash
# Step 2b: simulate frameshifted nanopore reads with Badread.
# Reads step-1 outputs (<base>.fa/.pos) READ-ONLY and writes
# <base>-fs.fa / <base>-fs.pos. NEVER touches step-1 files.
# Independent of step2-prepare.sh: both read only step-1 outputs and can
# run in any order. Downstream steps (3-4) combine the clean query HMMs
# from step2-prepare with these reads at benchmark time.
# Usage: ./step2-frameshift.sh [full|lite|full-over|lite-over|all] [--force]   (default: all)
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh

FORCE=0
VARIANT="all"
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        full|lite|full-over|lite-over|all) VARIANT="$arg" ;;
        *) echo "Usage: $0 [full|lite|full-over|lite-over|all] [--force]" >&2; exit 1 ;;
    esac
done

run_one() {
    local base="$1"
    local srcfa="$base.fa" srcpos="$base.pos"
    local outfa="$base-fs.fa" outpos="$base-fs.pos"
    local scratch="tmp-frameshift/$base"
    local reads="$scratch/reads.fastq" bed="$scratch/genes.bed"
    local mapped_tsv="$scratch/mapped.tsv" mapped_bed="$scratch/mapped.bed"

    # Structural ban: outputs must differ from step-1 inputs.
    if [[ "$outfa" == "$srcfa" || "$outpos" == "$srcpos" ]]; then
        echo "step2-frameshift: REFUSED: output names collide with step-1 inputs" >&2
        exit 1
    fi
    if [[ ! -f "$srcfa" || ! -f "$srcpos" ]]; then
        echo "step2-frameshift: missing step-1 outputs for $base, run step1 first" >&2
        exit 1
    fi
    if [[ -e "$outfa" || -e "$outpos" ]] && [[ "$FORCE" -eq 0 ]]; then
        echo "step2-frameshift: $outfa/$outpos already exist, skipping (use --force)"
        return 0
    fi
    if [[ ! -d "$BADREAD_DIR" ]]; then
        echo "step2-frameshift: Badread dir not found: $BADREAD_DIR" >&2
        exit 1
    fi
    if [[ ! -d "$BADREAD_HELPERS" ]]; then
        echo "step2-frameshift: Badread helpers dir not found: $BADREAD_HELPERS" >&2
        exit 1
    fi

    mkdir -p "$scratch"
    # .pos -> .bed from the pristine step-1 positives (never chained off .orig files)
    python3 "$BADREAD_HELPERS/pos_bed_converter.py" pos2bed "$srcpos" "$bed"
    python3 "$BADREAD_DIR/badread-runner.py" simulate \
        --reference "$srcfa" \
        --quantity "$FS_QUANTITY" \
        --error_model "$FS_ERROR_MODEL" --qscore_model "$FS_QSCORE_MODEL" \
        --identity "$FS_IDENTITY" \
        --chimeras "$FS_CHIMERAS" \
        --seed "$FS_SEED" \
        > "$reads"
    python3 "$BADREAD_HELPERS/map_gene_reads.py" \
        --reference "$srcfa" \
        --reads "$reads" \
        --genes "$bed" \
        --output "$mapped_tsv"
    tail -n +2 "$mapped_tsv" | awk -F'\t' 'BEGIN{OFS="\t"} {print $1, $3, $4, $2, "0", $5}' > "$mapped_bed"
    python3 "$BADREAD_HELPERS/pos_bed_converter.py" bed2pos \
        "$mapped_bed" "$scratch/mapped.pos"
    awk 'NR%4==1{print ">"substr($1,2)} NR%4==2{print}' "$reads" > "$outfa"
    cp "$scratch/mapped.pos" "$outpos"
    # Auto-repair: Badread UUIDs change each run, so any committed/cached
    # .fai is guaranteed stale (0 overlap -> bedtools skips all windows).
    # bedtools recreates it on demand, so always drop it here.
    rm -f "$outfa.fai" "$outfa.fai.old"
    echo "step2-frameshift: wrote $outfa and $outpos; step-1 $srcfa/$srcpos untouched"
}

case "$VARIANT" in
    full) run_one "$FULL_BASE" ;;
    lite) run_one "$LITE_BASE" ;;
    full-over) run_one "$FULL_OVER_BASE" ;;
    lite-over) run_one "$LITE_OVER_BASE" ;;
    all)  run_one "$FULL_BASE"
          run_one "$LITE_BASE"
          run_one "$FULL_OVER_BASE"
          run_one "$LITE_OVER_BASE" ;;
esac
