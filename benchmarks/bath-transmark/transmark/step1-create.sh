#!/bin/bash
# Step 1: create the transmark benchmark databases with transmark-create.
# Usage: ./step1-create.sh [full|lite|full-over|lite-over|all]   (default: all)
# full reuses the existing DB and is skipped unless --force is given,
# because regenerating it takes a very long time. lite-over is the fast
# overextension safety variant; full-over is the slow production one.
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh

VARIANT="${1:-all}"
FORCE=0
[[ "${2:-}" == "--force" ]] && FORCE=1

create_one() {
    local base="$1" args="$2"
    if [[ -f "$base.fa" && -f "$base.pos" && -f "$base.AA.msa" && "$FORCE" -eq 0 ]]; then
        echo "step1: $base.* already exist, skipping (use --force to regenerate)"
        return 0
    fi
    # shellcheck disable=SC2086
    ./transmark-create $args \
        -D "$CREATE_PFAM_SEED" "$base" \
        "$CREATE_DNA_TFILE" "$CREATE_AA_TFILE" \
        "$CREATE_BLAST_DIR" "$CREATE_EASEL_DIR" "$CREATE_BG_HMM" "$CREATE_FS_RATE"
}

case "$VARIANT" in
    full) create_one "$FULL_BASE" "$FULL_CREATE_ARGS" ;;
    lite) create_one "$LITE_BASE" "$LITE_CREATE_ARGS" ;;
    full-over) create_one "$FULL_OVER_BASE" "$FULL_OVER_CREATE_ARGS" ;;
    lite-over) create_one "$LITE_OVER_BASE" "$LITE_OVER_CREATE_ARGS" ;;
    all)  create_one "$FULL_BASE" "$FULL_CREATE_ARGS"
          create_one "$LITE_BASE" "$LITE_CREATE_ARGS"
          create_one "$FULL_OVER_BASE" "$FULL_OVER_CREATE_ARGS"
          create_one "$LITE_OVER_BASE" "$LITE_OVER_CREATE_ARGS" ;;
    *) echo "Usage: $0 [full|lite|full-over|lite-over|all] [--force]" >&2; exit 1 ;;
esac
