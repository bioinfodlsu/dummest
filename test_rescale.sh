#!/bin/bash
# test_rescale.sh — overflow-regime regression test for dummer's per-row
# dynamic rescaling (implementation: dummer-ovf.hh).
#
# What it covers:
#   1. herpes DOUBLE  (1359-aa profile x 10077-nt target): true ratio ~e^1507
#      overflows double; rescale path must report a finite score (~1.57e3),
#      E=0, at the true locus with no nan/inf.
#   2. medium DOUBLE  (500-codon slice, ratio ~e^476, finite in double):
#      rescale path must EXACTLY match the unrescaled-baseline score (539).
#   3-4. Same two cases with the float build (dummerl).
#
# Each output is checked by tools/test_rescale_check.py: the true (top) hit is
# fingerprinted (score/E/anchor/spans) and must be significant, and no other
# hit may be significant (E < FP_EVALUE, default 1e-5). Weak chance hits are
# allowed to vary run to run.
#
# Fixtures are generated deterministically under $FIXDIR (default
# /tmp/test_rescale). Binaries are rebuilt from current source first.
# Cold calibration takes a few minutes; warm re-runs are fast.
set -euo pipefail
cd "$(dirname "$0")"

FIXDIR="${TEST_RESCALE_TMPDIR:-/tmp/test_rescale}"
BUILD_DIR="${BUILD_DIR:-cmake-build-release}"
DUMMER_BIN="${DUMMER_BIN:-$BUILD_DIR/dummer}"
DUMMERL_BIN="${DUMMERL_BIN:-$BUILD_DIR/dummerl}"
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
python3 tools/test_rescale_fixtures.py "$FIXDIR"

# --- cases -------------------------------------------------------------------
# name | bin | cpus | fa | expected score | tol | E regex | anchor | spans
CASES=(
"herpes_D|$DUMMER_BIN|$NPROC|$FIXDIR/herpes.fa|1570|6|0|981,5942|TRAIN.Herpes_MCP,1,1358,+;herpes0,3003,4072,+"
"medium_D|$DUMMER_BIN|$NPROC|$FIXDIR/medium.fa|539|1|9\\..*e-216|531,2302|TRAIN.Herpes_MCP,426,506,+;medium0,1988,1516,+"
"herpes_L|$DUMMERL_BIN|$NPROC|$FIXDIR/herpes.fa|1569|6|0|-|TRAIN.Herpes_MCP,1,1358,+;herpes0,3003,4072,+"
"medium_L|$DUMMERL_BIN|$NPROC|$FIXDIR/medium.fa|539|1|9\\..*e-216|-|TRAIN.Herpes_MCP,426,506,+;medium0,1988,1516,+"
)

for spec in "${CASES[@]}"; do
    IFS='|' read -r name bin cpus fa score tol eregex anchor spans <<<"$spec"
    out="$FIXDIR/$name.out"
    log "run $name"
    python3 bin/pipeline2.py "$FIXDIR/herpes.hmm" MET.msa "$fa" "$cpus" \
        --max --dummer-bin "$bin" >"$out" 2>"$out.err"
    python3 tools/test_rescale_check.py "$out" "$score" "$tol" "$eregex" "$anchor" "$spans" "$FP_EVALUE" \
        || FAIL=1
done

if [[ $FAIL -ne 0 ]]; then
    echo "test_rescale.sh: FAILED"
    exit 1
fi
echo "test_rescale.sh: ALL PASS"
