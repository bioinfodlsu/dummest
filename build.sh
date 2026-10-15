#!/bin/bash
# build.sh — CMake configure + build (Ninja + ccache).
#
# Usage:
#   ./build.sh [cmake configure args...]
#
# Env:
#   SRC=<dir>        source dir              (default: .)
#   BUILD_DIR=<dir>  build dir               (default: build)
#   BUILD_TYPE=<t>   CMAKE_BUILD_TYPE        (default: Release)
#   JOBS=<n>         parallelism             (default: nproc)
#   TARGETS="t1 t2"  build targets           (default: all)
#
# Ninja and ccache are required. Example (portable container build):
#   SRC=. BUILD_DIR=/tmp/dbuild TARGETS="dummest_core dummestl_core" \
#       ./build.sh -DDUMMEST_PORTABLE=ON -DDUMMEST_SKIP_BIN_COPY=ON
set -euo pipefail

SRC="${SRC:-.}"
BUILD_DIR="${BUILD_DIR:-build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(nproc)}"

for tool in ninja ccache; do
    command -v "$tool" >/dev/null 2>&1 || { echo "$tool not found in PATH" >&2; exit 1; }
done

# Recreate a cache configured with a different generator (we always use Ninja).
if [[ -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    prev_gen="$(sed -n 's/^CMAKE_GENERATOR:INTERNAL=//p' "$BUILD_DIR/CMakeCache.txt")"
    if [[ -n "$prev_gen" && "$prev_gen" != "Ninja" ]]; then
        echo "==> generator changed ($prev_gen -> Ninja); recreating $BUILD_DIR" >&2
        rm -rf "$BUILD_DIR"
    fi
fi

echo "==> $SRC -> $BUILD_DIR ($BUILD_TYPE, Ninja + ccache)"
cmake -S "$SRC" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -G Ninja \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
    "$@"

args=(--build "$BUILD_DIR" --parallel "$JOBS")
if [[ -n "${TARGETS:-}" ]]; then
    # shellcheck disable=SC2206
    args+=(--target ${TARGETS})
fi
cmake "${args[@]}"
