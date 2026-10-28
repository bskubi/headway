#!/usr/bin/env bash
# Drops samples with too few counted reads from counts.tsv (featureCounts' layout: 6 annotation columns, then one
# column per sample).
set -euo pipefail

min_reads() {
  grep '^min_reads:' config.yaml | awk '{print $2}'
}

filter_samples() {
  local counts=$1 min
  min=$(min_reads)
  awk -v min="$min" '
    NR <= 2 { next }
    { for (j = 7; j <= NF; j++) total[j] += $j }
    END { for (j = 7; j <= NF; j++) if (total[j] >= min) print j }
  ' "$counts" > kept_columns.txt
}

filter_samples "${1:-counts.tsv}"
