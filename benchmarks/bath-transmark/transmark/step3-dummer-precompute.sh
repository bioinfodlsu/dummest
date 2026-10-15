#!/bin/bash
# Step 3 precompute for the dummer full benchmarks (used by step4-dummer.sh).
# Builds the expensive per-target artifacts once for the full / full-over
# frameshifted reads, so step4 runs can reuse them via pipeline.py
# --target-db-pad instead of rebuilding in a per-run tmpdir.
# Full-only: lite / lite-over targets are small and keep building per-run.
# Usage: ./step3-dummer-precompute.sh [full|full-over] [--force] [--no-gpu]
# Outputs (alongside the target FASTA):
#   <base>-fs.6frame.pfa      six-frame translation (seqkit translate -f 6 -F)
#   <base>-fs.targetDB_pad*   MMseqs DB prefix (mmseqs createdb)
# Mirrors pipeline.py per-run build:
#   seqkit translate -f 6 -F --threads <cpus> -o translated_6frame.pfa <fa>
#   mmseqs createdb <pfa> targetDB_pad --gpu <0|1> --threads <cpus>
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh

FORCE=0
GPU_FLAG=1
VARIANT="full"
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        --no-gpu) GPU_FLAG=0 ;;
        full) VARIANT="full" ;;
        full-over) VARIANT="full-over" ;;
        lite|lite-over|all) echo "step3-dummer-precompute: full/full-over only ($arg refused; lite/lite-over build per-run in tmpdir)" >&2; exit 1 ;;
        -h|--help) echo "Usage: $0 [full|full-over] [--force] [--no-gpu]" >&2; exit 0 ;;
        *) echo "Usage: $0 [full|full-over] [--force] [--no-gpu]" >&2; exit 1 ;;
    esac
done

if [[ "$VARIANT" == "full-over" ]]; then tbase="${FULL_OVER_BASE}-fs"; else tbase="${FULL_BASE}-fs"; fi
fa="$tbase.fa"
pfa="$tbase.6frame.pfa"
pad="$tbase.targetDB_pad"

[[ -f "$fa" ]] || { echo "step3-dummer-precompute: missing $fa -- run ./step2-frameshift.sh $VARIANT first" >&2; exit 1; }
command -v seqkit >/dev/null || { echo "step3-dummer-precompute: seqkit not found in PATH" >&2; exit 1; }
command -v mmseqs >/dev/null || { echo "step3-dummer-precompute: mmseqs not found in PATH" >&2; exit 1; }

if [[ -f "$pfa" && -f "$pad" && -f "$pad.index" && -f "$pad.dbtype" ]] && [[ "$FORCE" -eq 0 ]]; then
    # Auto-repair: prebuilt is keyed to exact Badread UUIDs; if reads are
    # newer than the artifacts they are guaranteed stale.
    if [[ "$fa" -nt "$pfa" || "$pfa" -nt "$pad" ]]; then
        echo "step3-dummer-precompute: $fa newer than $pfa/$pad*, stale, rebuilding (use --force to force anyway)" >&2
    else
        echo "step3-dummer-precompute: $pfa + $pad* already exist, skipping (use --force)"
        exit 0
    fi
fi

echo "step3-dummer-precompute: translating $fa -> $pfa (6-frame) ..."
seqkit seq -m 3 "$fa" | seqkit translate -f 6 -F --threads 24 > "$pfa"
# Inline assert: -F must emit frame= headers (mirrors pipeline.py seqkit translate -f 6 -F).
head -1 "$pfa" | grep -q 'frame=' || { echo "step3-dummer-precompute: ERROR: $pfa missing frame= headers (seqkit -F dropped?)" >&2; exit 1; }

# echo "step3-dummer-precompute: building MMseqs DB $pfa -> $pad ..."
mmseqs createdb "$pfa" "$pad" --gpu "$GPU_FLAG" --threads "$BENCH_CPUS"

echo "step3-dummer-precompute: wrote $pfa and $pad* for $tbase"
ls -lh "$pfa" "$pad"* | awk '{print "  " $0}'
