#!/bin/bash
# test_gpu.sh — CPU-vs-GPU MMseqs2 prefilter parity. Runs inside the image:
#   apptainer test --nv dummest.sif
# Skips without a GPU. Env: NPROC, TEST_GPU_TMPDIR (default /tmp/test_gpu).
set -euo pipefail
command -v nvidia-smi >/dev/null 2>&1 || { echo "SKIP: no NVIDIA GPU"; exit 0; }

NPROC="${NPROC:-$(nproc)}"
TMP="${TEST_GPU_TMPDIR:-/tmp/test_gpu}"
mkdir -p "$TMP"
HMM=/opt/dummest/samples/MET-test.hmm
MSA=/opt/dummest/samples/MET.msa
FA=/opt/dummest/samples/MET-target.fa
for label in cpu gpu; do
    flag=(--gpu off)
    [[ "$label" == gpu ]] && flag=(--gpu on)
    pipeline.py "$HMM" "$MSA" "$FA" "$NPROC" "${flag[@]}" \
        --output-fa "$TMP/$label.fa" --batch 1 >"$TMP/$label.log" 2>&1
done
python3 - "$TMP/cpu.fa" "$TMP/gpu.fa" <<'PY'
import sys

def load(path):
    d, key = {}, None
    for line in open(path):
        if line.startswith(">"):
            key = line[1:].strip()
            d[key] = []
        elif key is not None:
            d[key].append(line.strip())
    return {k: "".join(v) for k, v in d.items()}

cpu, gpu = load(sys.argv[1]), load(sys.argv[2])
if cpu != gpu:
    only_cpu = sorted(set(cpu) - set(gpu))
    only_gpu = sorted(set(gpu) - set(cpu))
    diff = [k for k in cpu if k in gpu and cpu[k] != gpu[k]]
    print(f"MISMATCH: cpu={len(cpu)} gpu={len(gpu)} "
          f"only_cpu={len(only_cpu)} only_gpu={len(only_gpu)} differing={len(diff)}",
          file=sys.stderr)
    for k in (only_cpu[:5] + only_gpu[:5] + diff[:5]):
        print(f"  {k}", file=sys.stderr)
    sys.exit(1)
if not cpu:
    print("EMPTY: no candidate regions extracted; test is vacuous", file=sys.stderr)
    sys.exit(1)
print(f"cpu == gpu ({len(cpu)} candidate region(s))")
PY
echo "test_gpu.sh: PASS"
