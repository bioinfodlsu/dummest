#!/bin/bash
cd "$(dirname "$0")/.."

source ~/.venv/bin/activate

mkdir -p qc

time bin/dummest samples/MET-test.hmm samples/MET-target.fa -S qc/scores.tsv

profiles=$(tail -n +2 qc/scores.tsv | cut -f2 | sort -u)
tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

specs=()
for prof in $profiles; do
    full_file="$tmpdir/${prof}.full.scores"
    filter_file="$tmpdir/${prof}.filter.scores"
    awk -F'\t' -v p="$prof" '$1=="full" && $2==p && $4=="mid" {print $5}' qc/scores.tsv > "$full_file"
    awk -F'\t' -v p="$prof" '$1=="filter" && $2==p && $4=="all" {print $5}' qc/scores.tsv > "$filter_file"
    full_count=$(awk -F'\t' -v p="$prof" '$1=="full" && $2==p && $4=="mid" && $5!="-inf" {print $5}' qc/scores.tsv | wc -l)
    filter_count=$(awk -F'\t' -v p="$prof" '$1=="filter" && $2==p && $4=="all" && $5!="-inf" {print $5}' qc/scores.tsv | wc -l)
    if [ "$full_count" -lt 5 ] && [ "$filter_count" -lt 5 ]; then
        echo "# Skipping ${prof}: only ${full_count} full and ${filter_count} filter finite scores"
        continue
    fi
    full_arg="$full_file"
    filter_arg="$filter_file"
    [ "$full_count" -lt 5 ] && full_arg=""
    [ "$filter_count" -lt 5 ] && filter_arg=""
    specs+=("${prof}:${full_arg}:${filter_arg}")
done
if [ ${#specs[@]} -eq 0 ]; then
    echo "# No families with finite scores"
    exit 1
fi
input_args=()
for s in "${specs[@]}"; do
    input_args+=(--input "$s")
done
rm -f qc/full_*.png qc/filter_*.png
python3 tools/plot_scores.py \
    --fit gumbel \
    --output "qc/scores.png" \
    "${input_args[@]}"
