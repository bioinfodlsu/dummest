#!/bin/bash
# test_overflow.sh — overflow regression test for dummest's row-rescale
# anti-overflow scheme (fast Float path with per-row 2^-64 dynamic
# rescaling; pair-domain ExpScore selection;
# implementation: dummest-ovf.hh, dummest-padded.hh).
#
# Cases: herpes (ratio ~e^1507, overflows double) and a 500-codon medium
# slice (ratio ~e^476, finite in double), each with the double and float
# builds. The herpes cases exercise the per-row rescale triggers and
# cum-folded promotion; the medium cases stay near static scale.
#
# Each output is checked by tools/test_overflow_check.py: the true (top) hit is
# fingerprinted (score/E/anchor/spans) and must be significant, and no other
# hit may be significant (E < FP_EVALUE, default 1e-5). Weak chance hits may
# vary run to run.
#
# Fixtures are generated deterministically under $FIXDIR (default
# /tmp/test_overflow); binaries are rebuilt first. Cold calibration takes a
# few minutes; warm re-runs are fast.
set -euo pipefail
cd "$(dirname "$0")"

FIXDIR="${TEST_OVERFLOW_TMPDIR:-/tmp/test_overflow}"
BUILD_DIR="${BUILD_DIR:-cmake-build-release}"
DUMMER_BIN="${DUMMER_BIN:-$BUILD_DIR/dummest}"
DUMMERL_BIN="${DUMMERL_BIN:-$BUILD_DIR/dummestl}"
FP_EVALUE="${FP_EVALUE:-1e-5}"
NPROC="${NPROC:-$(nproc)}"
FAIL=0

log() { echo "==> $*"; }

# --- source + build ----------------------------------------------------------
if [[ ! -f transmark-full.AA.hmm ]]; then
    echo "missing transmark-full.AA.hmm in repo root" >&2
    exit 1
fi
if [[ ! -d "$BUILD_DIR" ]]; then
    log "configuring $BUILD_DIR"
    cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
fi
log "building dummer + dummerl"
cmake --build "$BUILD_DIR" --target dummer dummerl -j "$(nproc)"

# --- fixtures ----------------------------------------------------------------
mkdir -p "$FIXDIR"
log "generating fixtures in $FIXDIR"
python3 tools/test_overflow_fixtures.py "$FIXDIR"

# --- cases -------------------------------------------------------------------
# name | bin | cpus | fa | expected score | tol | E regex | anchor | spans
CASES=(
 "herpes_D|$DUMMER_BIN|$NPROC|$FIXDIR/herpes.fa|2235.42|6|0|981,5942|TRAIN.Herpes_MCP,1,1358,+;herpes0,3003,4072,+"
 "medium_D|$DUMMER_BIN|$NPROC|$FIXDIR/medium.fa|749.37|1|9\\..*e-216|531,2302|TRAIN.Herpes_MCP,426,506,+;medium0,1988,1516,+"
 "herpes_L|$DUMMERL_BIN|$NPROC|$FIXDIR/herpes.fa|2235.42|6|0|-|TRAIN.Herpes_MCP,1,1358,+;herpes0,3003,4072,+"
 "medium_L|$DUMMERL_BIN|$NPROC|$FIXDIR/medium.fa|749.37|1|9\\..*e-216|-|TRAIN.Herpes_MCP,426,506,+;medium0,1988,1516,+"
)

for spec in "${CASES[@]}"; do
    IFS='|' read -r name bin cpus fa score tol eregex anchor spans <<<"$spec"
    out="$FIXDIR/$name.out"
    log "run $name"
    python3 bin/pipeline.py "$FIXDIR/herpes.hmm" MET.msa "$fa" "$cpus" \
        --max --dummest-bin "$bin" >"$out" 2>"$out.err"
    python3 tools/test_overflow_check.py "$out" "$score" "$tol" "$eregex" "$anchor" "$spans" "$FP_EVALUE" \
        || FAIL=1
done

if [[ $FAIL -ne 0 ]]; then
    echo "test_overflow.sh: FAILED"
    exit 1
fi
echo "test_overflow.sh: ALL PASS"
