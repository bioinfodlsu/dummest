#!/bin/bash
# Step 2.5: rename duplicate query IDs in frameshifted *-fs.pos files.
# Step2 Badread simulation embeds the same query (e.g. NQRA/20) in many
# reads, so col1 is no longer unique (lite: 994 lines / 189 unique).
# Downstream tools (coverage/mer/rocplot/idpositives) assume col1 is a
# unique Fam/N key, which caused cross-read merges and negative cover.
# This script renumbers col1 per family (Fam/1..k, deterministic) so
# every embedding is unique, keeping Fam/N numeric format required by
# transmark-idpositives.pl (^\S+\/\d+$). Family prefix, col2 (UUID),
# col3, coords untouched. Emits <base>-fs.rename.tsv mapping
# new -> old+target for collapsed per-gene reporting.
# Reads <base>-fs.pos (from step2-frameshift.sh) and rewrites it in
# place. Run once per benchmark, before step3/step4. Idempotent guard
# refuses to run twice without --force.
# Usage: ./step-2.5-rename-fs.sh [full|lite|full-over|lite-over|all] [--force]   (default: all)
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh

FORCE=0
VARIANT="all"
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        full|lite|full-over|lite-over|all) VARIANT="$arg" ;;
        *) echo "Usage: $0 [full|lite|full-over|lite-over|all] [--force]" >&2; exit 1 ;;
    esac
done

rename_one() {
    local base="$1"
    local pos="$base-fs.pos"
    local map="$base-fs.rename.tsv"
    if [[ ! -f "$pos" ]]; then
        echo "step-2.5: missing $pos, run ./step2-frameshift.sh first" >&2
        exit 1
    fi
    if [[ -e "$map" ]] && [[ "$FORCE" -eq 0 ]]; then
        echo "step-2.5: $map already exists, skipping (use --force)"
        return 0
    fi
    python3 - "$pos" "$map" <<'EOF'
import sys
from collections import defaultdict
pos, mapp = sys.argv[1], sys.argv[2]
rows = []
with open(pos) as fh:
    for line in fh:
        if not line.strip() or line.startswith('#'):
            continue
        p = line.split()
        # col1=Fam/idx col2=target col3=src col4/5=coords
        fam_full, target = p[0], p[1]
        assert '/' in fam_full, line
        fam, _old = fam_full.rsplit('/', 1)
        assert _old.isdigit(), line
        assert '/' not in fam, line
        rows.append((fam, line, p))
# deterministic: per family, sort by (old idx, target, coords)
byfam = defaultdict(list)
for fam, line, p in rows:
    byfam[fam].append((line, p))
out_lines = []
map_lines = []
for fam in sorted(byfam):
    entries = sorted(byfam[fam], key=lambda t: (t[1][0], t[1][1], t[1][3], t[1][4]))
    for i, (line, p) in enumerate(entries, 1):
        new = "%s/%d" % (fam, i)
        out_lines.append("%-35s %-36s %-35s %8s %8s\n" % (new, p[1], p[2], p[3], p[4]))
        map_lines.append("%s\t%s\t%s\t%s\t%s\t%s\n" % (new, p[0], p[1], p[2], p[3], p[4]))
with open(pos + '.new', 'w') as fh:
    fh.writelines(out_lines)
with open(mapp + '.new', 'w') as fh:
    fh.writelines(map_lines)
EOF
    mv "$pos.new" "$pos"
    mv "$map.new" "$map"
    echo "step-2.5: renamed $pos ($(( $(wc -l < "$pos") )) embeddings) + wrote $map"
}

case "$VARIANT" in
    full) rename_one "$FULL_BASE" ;;
    lite) rename_one "$LITE_BASE" ;;
    full-over) rename_one "$FULL_OVER_BASE" ;;
    lite-over) rename_one "$LITE_OVER_BASE" ;;
    all)  rename_one "$FULL_BASE"
          rename_one "$LITE_BASE"
          rename_one "$FULL_OVER_BASE"
          rename_one "$LITE_OVER_BASE" ;;
esac
