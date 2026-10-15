#!/bin/bash
# Step 2: build query HMMs from a step1 benchmark.
# Independent of step2-frameshift.sh: both read only step-1 outputs and
# can run in any order. Downstream steps (3-4) combine the clean query
# HMMs built here with frameshifted reads at benchmark time.
# Usage: ./step2-prepare.sh [full|lite|full-over|lite-over|all]   (default variant: all)
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh

for arg in "$@"; do
    case "$arg" in
        --clean) echo "step2-prepare: --clean is now a no-op (clean queries are always built)" >&2 ;;
        --fs) echo "step2-prepare: --fs is deprecated (frameshift reads are consumed downstream, not here)" >&2 ;;
    esac
done
if [[ "${1:-}" == "--clean" || "${1:-}" == "--fs" ]]; then shift; fi
VARIANT="${1:-all}"

prepare_one() {
    ./prepare_queries_and_targets.sh "$1" . \
        "$PREP_HMMER_DIR" "$PREP_BATH1_DIR" "$PREP_BATH2_DIR" \
        "$PREP_DUMMER_DIR" "$PREP_CPUS"
}

case "$VARIANT" in
    full) prepare_one "$FULL_BASE" ;;
    lite) prepare_one "$LITE_BASE" ;;
    full-over) prepare_one "$FULL_OVER_BASE" ;;
    lite-over) prepare_one "$LITE_OVER_BASE" ;;
    all)  prepare_one "$FULL_BASE"
          prepare_one "$LITE_BASE"
          prepare_one "$FULL_OVER_BASE"
          prepare_one "$LITE_OVER_BASE" ;;
    *) echo "Usage: $0 [full|lite|full-over|lite-over|all]" >&2; exit 1 ;;
esac
