#!/bin/bash
# Export a generated/temp-file-free copy of transmark into DEST.
# Uses git's index + .gitignore as the single source of truth: tracked
# files plus untracked source files, excluding all benchmark data,
# results, logs, scratch, build artifacts and local junk.
# Usage: ./export-clean.sh <destination-dir>
set -euo pipefail
cd "$(dirname "$0")"

DEST="${1:?Usage: $0 <destination-dir>}"
[[ -e "$DEST" ]] && { echo "export-clean: refusing, $DEST already exists" >&2; exit 1; }

git rev-parse --show-toplevel >/dev/null 2>&1 \
    || { echo "export-clean: not inside a git repository" >&2; exit 1; }
ROOT="$(git rev-parse --show-toplevel)"
DEST_ABS="$(realpath -m "$DEST")"
case "$DEST_ABS/" in
    "$ROOT"/*) echo "export-clean: refusing, destination is inside the repo" >&2; exit 1 ;;
esac

mkdir -p "$DEST"
git ls-files -z --cached --others --exclude-standard -- . \
    | tar --null --no-recursion -cf - -T - \
    | tar -xf - -C "$DEST"
echo "export-clean: exported $(find "$DEST" -type f | wc -l) files to $DEST"
