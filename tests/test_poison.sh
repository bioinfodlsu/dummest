#!/bin/bash
# test_poison.sh — stale-scratch debug mode for dummest.
#
# Builds normal + DUMMEST_POISON binaries (Debug, with ASan/UBSan + asserts)
# and runs a small DP workload through both. The poison build fills
# uninitialized scratch (W1 via resizeNoFill, Y0_curr) with quiet-NaN and
# asserts on first read of a poisoned cell (see dummest-poison.hh).
#
# Success = poison run finishes with output identical to the normal run.
# Any assert / NaN leak means a SIMD lane (or scalar path) read stale,
# previous-batch, or out-of-bounds data.
#
# Integration workload is intentionally batch-sensitive: varying
# DUMMEST_MAX_LANES forces partial batches (inactive lanes),
# DUMMEST_TRUNC_LANES=1 duplicates lane 0 to full width with staggered prefix
# lengths (mixed active lengths from a single-sequence input), and the
# length-sorted pipeline reuses each thread's DPScratch across batches with
# shrinking/growing dpWidth -- the exact reuse that hides stale reads.
#
# Usage: tests/test_poison.sh [--keep-build]
# Env: BUILD_DIR (default cmake-build-poison), NPROC, FIXDIR fixtures.
set -euo pipefail
cd "$(dirname "$0")/.."

FIXDIR="$PWD/tests/fixtures"
if [[ ! -f "$FIXDIR/herpes.hmm" ]]; then
    echo "missing fixtures in $FIXDIR (need herpes.hmm)" >&2
    exit 1
fi
if [[ ! -f "$FIXDIR/herpes.fa" ]]; then
    echo "missing fixtures in $FIXDIR (need herpes.fa)" >&2
    exit 1
fi

# Poison binaries must not share a calibration cache with normal builds (the
# binary hash is part of the cache key, so they naturally diverge; each side
# below uses its own dir under its build tree).
export DUMMEST_CACHE_DIR="${DUMMEST_CACHE_DIR:-${TMPDIR:-/tmp}/.dummest-cache-poison-test}"
mkdir -p "$DUMMEST_CACHE_DIR"

BUILD_DIR="${BUILD_DIR:-cmake-build-poison}"
NORM_DIR="${NORM_DIR:-cmake-build-poison-normal}"
NPROC="${NPROC:-$(nproc)}"
FAIL=0
log() { echo "==> $*"; }

build_one() { # $1=dir $2=poison(ON/OFF) $3=label
    local dir="$1" poison="$2" label="$3"
    if [[ ! -d "$dir" ]]; then
        log "configuring $dir ($label)"
        cmake -S . -B "$dir" -DCMAKE_BUILD_TYPE=Debug -DDUMMEST_POISON="$poison" \
            -DDUMMEST_SKIP_BIN_COPY=ON >"$dir.configure.log" 2>&1
    else
        log "reconfiguring $dir ($label)"
        cmake -S . -B "$dir" -DCMAKE_BUILD_TYPE=Debug -DDUMMEST_POISON="$poison" \
            -DDUMMEST_SKIP_BIN_COPY=ON >"$dir.configure.log" 2>&1
    fi
    log "building $label"
    cmake --build "$dir" --target dummest dummestl -j "$NPROC" >"$dir.build.log" 2>&1
}

build_one "$NORM_DIR" OFF "normal"
build_one "$BUILD_DIR" ON "poison"

# Unit tests: normal + poison (poison sentinel shape checks in test_batch_pairs).
run_unit() { # $1=dir $2=label $3=extra-cxxflags
    local dir="$1" label="$2"
    log "unit tests ($label)"
    local kok="$PWD/kokkos-5.1.1" bldkok="$PWD/$dir/kokkos-5.1.1"
    local defs=""
    if [[ "$label" == poison* ]]; then defs="-DDUMMEST_POISON"; fi
    # shellcheck disable=SC2086
    g++ -std=c++20 -O1 -g $defs -march=native -I"$PWD" \
        -I"$kok/simd/src" -I"$kok/core/src" -I"$bldkok" \
        -I"$kok/tpls/desul/include" -I"$bldkok/core/src" \
        -I"$kok/tpls/mdspan/include" \
        tests/test_batch_pairs.cc -o "$dir/test_batch_pairs_$label" \
        >"$dir.unit_build.log" 2>&1
    "$dir/test_batch_pairs_$label" >"$dir.unit.log" 2>&1 || { FAIL=1; tail -20 "$dir.unit.log"; }
    tail -2 "$dir.unit.log"
}
run_unit "$NORM_DIR" normal
run_unit "$BUILD_DIR" poison

# Integration: same workload, both binaries, several lane counts.
# Small workload: herpes profile + medium target, --max (slow wave, exercises
# backward+forward + traceback), few threads so batches reuse scratch
# repeatedly. --trials/--length keep cold calibration fast even under
# Debug+ASan (deterministic RNG, so normal vs poison outputs must match).
# NOTE: each binary uses its own DUMMEST_CACHE_DIR (binary hash is part of the
# cache key), so both runs calibrate from scratch under their own build.
TRIALS="${TRIALS:-50}"
CALIB_LEN="${CALIB_LEN:-500}"
compare_runs() { # $1=lanes $2=trunc(0/1, default 0)
    local lanes="$1" trunc="${2:-0}"
    local tag="L${lanes}"
    if [[ "$trunc" == 1 ]]; then tag="L${lanes}_trunc"; fi
    local outN="$BUILD_DIR/poison_test_normal_${tag}.out"
    local outP="$BUILD_DIR/poison_test_poison_${tag}.out"
    log "run lanes=$lanes trunc=$trunc (normal vs poison)"
    DUMMEST_CACHE_DIR="$NORM_DIR/.dummest-cache" DUMMEST_MAX_LANES="$lanes" DUMMEST_TRUNC_LANES="$trunc" python3 bin/pipeline.py \
        "$FIXDIR/herpes.hmm" samples/MET.msa "$FIXDIR/medium.fa" 2 \
        --max --trials "$TRIALS" --length "$CALIB_LEN" \
        --dummest-bin "$PWD/$NORM_DIR/dummestl" >"$outN" 2>"$outN.err" || FAIL=1
    DUMMEST_CACHE_DIR="$BUILD_DIR/.dummest-cache" DUMMEST_MAX_LANES="$lanes" DUMMEST_TRUNC_LANES="$trunc" python3 bin/pipeline.py \
        "$FIXDIR/herpes.hmm" samples/MET.msa "$FIXDIR/medium.fa" 2 \
        --max --trials "$TRIALS" --length "$CALIB_LEN" \
        --dummest-bin "$PWD/$BUILD_DIR/dummestl" >"$outP" 2>"$outP.err" || FAIL=1
    # No NaN/inf tokens allowed in either output.
    if grep -Ei 'nan|inf' "$outP" >/dev/null; then
        echo "FAIL lanes=$lanes trunc=$trunc: nan/inf token in poison output"; FAIL=1
    fi
    # Score lines must match (order may differ: batches print as they finish;
    # sort before diff). Both binaries truncate identically (pure function of
    # L and lane), so truncated outputs stay comparable.
    if ! diff <(grep '^a ' "$outN" | sort) <(grep '^a ' "$outP" | sort) >"$BUILD_DIR/diff_${tag}.txt"; then
        echo "FAIL lanes=$lanes trunc=$trunc: poison output differs (see $BUILD_DIR/diff_${tag}.txt)"; FAIL=1
    else
        echo "ok lanes=$lanes trunc=$trunc: $(grep -c '^a ' "$outP") hit(s), outputs identical"
    fi
}

for L in 1 2 8; do
    compare_runs "$L" 0 || true
done
# Truncated-lane legs: lane 0 duplicated to full width with staggered prefix
# lengths (DUMMEST_TRUNC_LANES=1); exercises dpWidth=max with many past-end
# cutoffs from a single-sequence input. Slicing still follows batchStride,
# so cover both multi-job (lanes=1) and single-job (lanes=8) shapes.
for L in 1 8; do
    compare_runs "$L" 1 || true
done

if [[ $FAIL -ne 0 ]]; then
    echo "test_poison.sh: FAILED"
    exit 1
fi
echo "test_poison.sh: ALL PASS"
