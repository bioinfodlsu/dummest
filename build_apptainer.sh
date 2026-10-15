#!/bin/bash
# build_apptainer.sh — build the DUMMEST Apptainer image.
#
# Both binaries are built on the host and copied into a slim runtime image (no
# compiler, no CUDA toolkit):
#   - DUMMEST, portable baseline ISA (-DDUMMEST_PORTABLE=ON: x86-64-v3/AVX2 on
#     amd64, armv8-a/NEON on aarch64)
#   - the custom MMseqs2 prefilter fork: CUDA-enabled when the CUDA toolkit is
#     present at build time (statically links libcudart; runtime needs only
#     libcuda.so.1, injected by `apptainer --nv`), CPU-only otherwise
# The host distro/arch must match the image base (Ubuntu noble).
#
# The MMseqs2 fork (branch prefilter) is bundled as the MMseqs2/ submodule;
# initialize it with: git submodule update --init --recursive
#
# Required on PATH:
#   ninja, ccache (sudo apt install ninja-build ccache)
#   (plus cmake and cargo; nvcc too when building with CUDA)
# Optional:
#   MMSEQS_SRC=<path>           (default: the bundled MMseqs2/ submodule)
#   SIF=<output image>          (default: dummest.sif)
#   CUDA=auto|1|0               (default: auto; CUDA when nvcc is found, via
#                               $CUDACXX/PATH//usr/local/cuda — arch-agnostic)
#   CUDA_ARCHS=<semicolon list> (default: 75-real;80-real;86-real;89-real;90-real;120-real;
#                               override for non-x86 GPUs, e.g. Grace-Hopper/Jetson)
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
MMSEQS_SRC="${MMSEQS_SRC:-$PWD/MMseqs2}"

command -v apptainer >/dev/null 2>&1 || {
    echo "apptainer not found in PATH" >&2
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
[[ -f "$MMSEQS_SRC/CMakeLists.txt" ]] || {
    echo "MMseqs2 fork not found at $MMSEQS_SRC" >&2
    echo "  set MMSEQS_SRC, or run: git submodule update --init --recursive" >&2
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

# --- helper ---------------------------------------------------------------
need() {
    command -v "$1" >/dev/null 2>&1 || { echo "$1 not found ($2)" >&2; exit 1; }
}

# --- CUDA is orthogonal to the host arch (the toolkit targets aarch64 too, ---
# --- e.g. Grace-Hopper/Jetson); select it by toolchain availability.      ---
CUDA="${CUDA:-auto}"
if [[ "$CUDA" == "auto" ]]; then
    if [[ -n "${CUDACXX:-}" && -x "${CUDACXX}" ]]; then
        CUDA=1
    elif command -v nvcc >/dev/null 2>&1; then
        CUDA=1
    elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
        CUDA=1
        export CUDACXX=/usr/local/cuda/bin/nvcc
    else
        CUDA=0
    fi
fi
[[ "$CUDA" == "0" || "$CUDA" == "1" ]] || {
    echo "CUDA must be auto, 1, or 0 (got: $CUDA)" >&2
    exit 1
}
if [[ "$CUDA" == "1" ]]; then
    need "${CUDACXX:-nvcc}" "install the CUDA toolkit"
fi

mkdir -p "$CACHE"

# --- MMseqs2 fork (host build, cached) ---
MMSEQS_BIN_PATH="$CACHE/mmseqs"
if [[ "${REBUILD_MMSEQS:-0}" == "1" || ! -x "$MMSEQS_BIN_PATH" ]]; then
    echo "==> building MMseqs2 fork ($([[ "$CUDA" == "1" ]] && echo "CUDA archs: $CUDA_ARCHS" || echo "CPU-only"))"
    need cargo "the MMseqs2 build requires Rust"

    cuda_args=(-DENABLE_CUDA="$CUDA")
    if [[ "$CUDA" == "1" ]]; then
        cuda_args+=(-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS")
    fi

    bdir="$(mktemp -d "${TMPDIR:-/tmp}/dummest-mmseqs-build.XXXXXX")"
    SRC="$MMSEQS_SRC" BUILD_DIR="$bdir/build" TARGETS=mmseqs \
        ./build.sh "${cuda_args[@]}"

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
if [[ "${REBUILD_DUMMEST:-0}" == "1" || ! -x "$DUMMEST_CACHE/dummestl_core" ]]; then
    echo "==> building DUMMEST (portable)"
    BUILD_DIR="$DUMMEST_CACHE" TARGETS="dummest_core dummestl_core dummest-build" \
        ./build.sh -DDUMMEST_PORTABLE=ON -DDUMMEST_SKIP_BIN_COPY=ON
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
install -m755 "$DUMMEST_CACHE/dummest_core" "$DUMMEST_CACHE/dummestl_core" "$DUMMEST_CACHE/dummest-build" "$STAGE_DIR/"
install -m755 bin/dummest "$STAGE_DIR/"
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
