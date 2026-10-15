#!/usr/bin/env bash
# run_dummer_only.sh — DUMMER warmup + DUMMER leg (wraps run_ncbi_comparison.sh).
#
# Paths/params come from config.sh (see config.example.sh).
# Usage:
#   ./run_dummer_only.sh
#   N_READS=10 BASE=/tmp/dummer_test ./run_dummer_only.sh
set -euo pipefail

BASE=${BASE:-$HOME/ncbi_runs/dummer_only}
DUM_NOSEED=${DUM_NOSEED:-$BASE/dummer}
export BASE DUM_NOSEED
exec "$(dirname "$0")/run_ncbi_comparison.sh" --only dummer "$@"
