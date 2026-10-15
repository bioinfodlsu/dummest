#!/usr/bin/env bash
# lib.sh — shared helpers for the run wrappers.

# Source config.sh if present (override the path with CONFIG=...).
load_config() {
    local repo_dir="$1"
    local config="${CONFIG:-$repo_dir/config.sh}"
    [[ -f "$config" ]] && source "$config"
}

# Resolve mmseqs from MMSEQS_BIN or ~/bin; require seqtk and seqkit.
require_cmds() {
    command -v mmseqs >/dev/null \
        || export PATH="$(dirname "${MMSEQS_BIN:-$HOME/bin/mmseqs}"):$PATH"
    command -v seqtk  >/dev/null || { echo "seqtk not on PATH" >&2; exit 1; }
    command -v seqkit >/dev/null || { echo "seqkit not on PATH" >&2; exit 1; }
}

# DUMMER calibration warmup; -T 24 is independent of $CPU.
dummer_warmup() {
    local engine="$1" hmmdb="$2" log="$3"
    mkdir -p "$(dirname "$log")"
    echo "--- DUMMER warmup ---"
    "$engine" -T 24 "$hmmdb" --warmup &> "$log"
}
