set -euo pipefail
trap 'echo "run-full-over: FAILED at line $LINENO: $BASH_COMMAND (exit $?)" >&2' ERR
cd "$(dirname "$0")"

V=full-over
TAG_FS=fs
TAG_CLEAN=clean

echo "=== [1/7] step1-create $V ===" >&2
./step1-create.sh "$V"
echo "=== [2/7] step2-prepare $V ===" >&2
./step2-prepare.sh "$V"
echo "=== [3/7] step2-frameshift + rename $V (slow Badread) ===" >&2
./step2-frameshift.sh "$V"
./step-2.5-rename-fs.sh "$V"
echo "=== [4/7] dummer precompute $V ===" >&2
./step3-dummer-precompute.sh "$V"

echo "=== [5/7] BATH $V ===" >&2
{ time ./step3-bathsearch.sh "$V"; } &> "step3-bathsearch-$V-$TAG_FS.log"
{ time ./step3-bathsearch.sh --no-fs "$V"; } &> "step3-bathsearch-no-fs-$V-$TAG_FS.log"
# { time ./step3-bathsearch.sh --clean "$V"; } &> "step3-bathsearch-$V-$TAG_CLEAN.log"
# { time ./step3-bathsearch.sh --no-fs --clean "$V"; } &> "step3-bathsearch-no-fs-$V-$TAG_CLEAN.log"

echo "=== [6/7] DUMMER $V ===" >&2
{ time ./step4-dummer.sh "$V"; } &> "step4-dummer-$V-$TAG_FS.log"
# { time ./step4-dummer.sh "$V" --gapped-pvalue 0.005; } &> "step4-dummer-$V-$TAG_FS-fast.log"
# { time ./step4-dummer.sh --clean "$V"; } &> "step4-dummer-$V-$TAG_CLEAN.log"

echo "=== [7/7] LAST $V ===" >&2
{ time ./step5-last.sh prepare "$V"; } &> "step5-last-prepare-$V-$TAG_FS.log"
{ time ./step5-last.sh run "$V"; } &> "step5-last-$V-$TAG_FS.log"

echo "=== figures $V (fs) ===" >&2
./step-last-generate-figures.sh "$V"
echo "run-full-over: done. Results in $V(-fs)-results* + modern_rocplot-$V-fs.png" >&2
