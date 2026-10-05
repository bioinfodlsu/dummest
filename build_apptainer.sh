#!/bin/bash
# build_apptainer.sh — build the DUMMEST Apptainer image.
#
# Both binaries are built on the host and copied into a slim runtime image (no
# compiler, no CUDA toolkit):
#   - DUMMEST, portable AVX2 (-DDUMMEST_PORTABLE=ON)
#   - the custom GPU MMseqs2 fork (statically links libcudart; runtime needs
#     only libcuda.so.1, injected by `apptainer --nv`)
# The host distro/arch must match the image base (Ubuntu noble, amd64).
#
# Required:
#   MMSEQS_SRC=<path to the MMseqs2 fork source tree>
#   ninja, ccache on PATH (sudo apt install ninja-build ccache)
#   (plus cmake, nvcc, and cargo when a rebuild is actually needed)
# Optional:
#   SIF=<output image>          (default: dummest.sif)
#   CUDA_ARCHS=<semicolon list> (default: 75-real;80-real;86-real;89-real;90-real;120-real)
#   REBUILD_MMSEQS=1            force a fresh mmseqs build
#   REBUILD_DUMMEST=1           force a fresh DUMMEST build
#   APT_CACHE=1                 reuse the host apt caches (same distro/arch only)
#   MMSEQS_BUILD_CACHE=<dir>    (default: ~/.cache/dummest-apptainer)
#   JOBS=<n>                    (default: nproc)
#   NV=0                        build without --nv (default: --nv when nvidia-smi is present,
#                               so the build-time test covers the GPU leg)
#   SKIP_DISTRO_CHECK=1         skip the host/image distro check
# Extra arguments are forwarded to `apptainer build`.
set -euo pipefail
cd "$(dirname "$0")"

SIF="${SIF:-dummest.sif}"
CUDA_ARCHS="${CUDA_ARCHS:-75-real;80-real;86-real;89-real;90-real;120-real}"
CACHE="${MMSEQS_BUILD_CACHE:-$HOME/.cache/dummest-apptainer}"
JOBS="${JOBS:-$(nproc)}"
MMSEQS_SRC="${MMSEQS_SRC:-}"

command -v apptainer >/dev/null 2>&1 || {
    echo "apptainer not found in PATH" >&2
    exit 1
}
command -v ninja >/dev/null 2>&1 || {
    echo "ninja not found in PATH; install with: sudo apt install ninja-build" >&2
    exit 1
}
command -v ccache >/dev/null 2>&1 || {
    echo "ccache not found in PATH; install with: sudo apt install ccache" >&2
    exit 1
}
[[ -f dummest.def ]] || {
    echo "dummest.def not found" >&2
    exit 1
}
[[ -f kokkos-5.1.1/CMakeLists.txt ]] || {
    echo "kokkos-5.1.1 submodule not initialized; run: git submodule update --init --recursive" >&2
    exit 1
}
if [[ -z "$MMSEQS_SRC" ]]; then
    echo "MMSEQS_SRC is required (path to the MMseqs2 fork source tree); e.g." >&2
    echo "  MMSEQS_SRC=/path/to/MMseqs2 ./build_apptainer.sh" >&2
    exit 1
fi
[[ -f "$MMSEQS_SRC/CMakeLists.txt" ]] || {
    echo "$MMSEQS_SRC/CMakeLists.txt not found" >&2
    exit 1
}

# Binaries are built on the host and run in the image, so the host must be
# Ubuntu 24.04 to match the image base.
if [[ "${SKIP_DISTRO_CHECK:-0}" != "1" ]]; then
    host_ver="$(. /etc/os-release; echo "$VERSION_ID")"
    [[ "$host_ver" == "24.04" ]] || {
        echo "ERROR: host Ubuntu $host_ver != 24.04 (image base ubuntu:24.04)" >&2
        echo "Build on Ubuntu 24.04 or set SKIP_DISTRO_CHECK=1." >&2
        exit 1
    }
fi

mkdir -p "$CACHE"

# --- build flags: Ninja + ccache, always. ---
GEN=(-G Ninja); GEN_NAME=Ninja
LAUNCH=(-DCMAKE_C_COMPILER_LAUNCHER=ccache
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
        -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache)
echo "==> generator: Ninja (ccache enabled)"

need() {
    command -v "$1" >/dev/null 2>&1 || { echo "$1 not found ($2)" >&2; exit 1; }
}

# --- MMseqs2 fork (host build, cached) ---
MMSEQS_BIN_PATH="$CACHE/mmseqs"
if [[ "${REBUILD_MMSEQS:-0}" == "1" || ! -x "$MMSEQS_BIN_PATH" ]]; then
    echo "==> building MMseqs2 fork (CUDA archs: $CUDA_ARCHS)"
    need cmake "install the CMake build tool"
    need cargo "the MMseqs2 build requires Rust"
    if [[ -z "${CUDACXX:-}" && -x /usr/local/cuda/bin/nvcc ]]; then
        export CUDACXX=/usr/local/cuda/bin/nvcc
    fi
    need "${CUDACXX:-nvcc}" "install the CUDA toolkit"

    bdir="$(mktemp -d "${TMPDIR:-/tmp}/dummest-mmseqs-build.XXXXXX")"
    cmake -S "$MMSEQS_SRC" -B "$bdir/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_CUDA=1 \
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS" \
        "${GEN[@]}" "${LAUNCH[@]}"
    cmake --build "$bdir/build" --parallel "$JOBS" --target mmseqs

    built="$bdir/build/src/mmseqs"
    [[ -x "$built" ]] || { echo "mmseqs binary not found after build ($built)" >&2; exit 1; }
    install -m755 "$built" "$MMSEQS_BIN_PATH"
    rm -rf "$bdir"
    echo "==> staged $MMSEQS_BIN_PATH"
else
    echo "==> reusing cached mmseqs: $MMSEQS_BIN_PATH (REBUILD_MMSEQS=1 to rebuild)"
fi

# --- DUMMEST (host build, portable, cached) ---
DUMMEST_CACHE="$CACHE/dummest-build"
if [[ "${REBUILD_DUMMEST:-0}" == "1" || ! -x "$DUMMEST_CACHE/dummestl" ]]; then
    echo "==> building DUMMEST (portable AVX2)"
    need cmake "install the CMake build tool"
    if [[ -f "$DUMMEST_CACHE/CMakeCache.txt" ]]; then
        prev_gen="$(sed -n 's/^CMAKE_GENERATOR:INTERNAL=//p' "$DUMMEST_CACHE/CMakeCache.txt")"
        if [[ -n "$prev_gen" && "$prev_gen" != "$GEN_NAME" ]]; then
            echo "==> generator changed ($prev_gen -> $GEN_NAME); reconfiguring $DUMMEST_CACHE"
            rm -rf "$DUMMEST_CACHE"
        fi
    fi
    cmake -S "$PWD" -B "$DUMMEST_CACHE" \
        -DCMAKE_BUILD_TYPE=Release \
        -DDUMMEST_PORTABLE=ON \
        -DDUMMEST_SKIP_BIN_COPY=ON \
        "${GEN[@]}" "${LAUNCH[@]}"
    cmake --build "$DUMMEST_CACHE" --parallel "$JOBS" \
        --target dummest dummestl dummest-build
    echo "==> staged DUMMEST binaries in $DUMMEST_CACHE"
else
    echo "==> reusing cached DUMMEST: $DUMMEST_CACHE (REBUILD_DUMMEST=1 to rebuild)"
fi

# --- stage everything for the image ---
# Binaries go to .apptainer-stage/ and the repo payload to
# .apptainer-stage/payload/; dummest.def copies both in via %files.
STAGE_DIR="$PWD/.apptainer-stage"
trap 'rm -rf "$STAGE_DIR"' EXIT
rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR/payload"
install -m755 "$DUMMEST_CACHE/dummest" "$DUMMEST_CACHE/dummestl" "$DUMMEST_CACHE/dummest-build" "$STAGE_DIR/"
install -m755 bin/pipeline.py "$STAGE_DIR/"
install -m755 "$MMSEQS_BIN_PATH" "$STAGE_DIR/mmseqs"
git ls-files -z --cached --others --exclude-per-directory=.gitignore \
    | while IFS= read -r -d '' f; do [ -f "$f" ] && printf '%s\0' "$f"; done \
    | tar --null -T - -cf - | tar -C "$STAGE_DIR/payload" -xf -

# --- build the image ---
fakeroot=()
[[ "$(id -u)" -eq 0 ]] || fakeroot=(--fakeroot)

nv_args=()
if [[ "${NV:-1}" != "0" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    nv_args=(--nv)
fi

binds=()
if [[ "${APT_CACHE:-0}" == "1" ]]; then
    echo "==> reusing host apt caches (same distro/arch only)"
    binds+=(--bind /var/cache/apt/archives:/var/cache/apt/archives)
    binds+=(--bind /var/lib/apt/lists:/var/lib/apt/lists)
fi

echo "==> building $SIF"
apptainer build --force "${fakeroot[@]}" "${binds[@]}" "${nv_args[@]}" "$SIF" dummest.def "$@"
echo "==> done: $SIF"
