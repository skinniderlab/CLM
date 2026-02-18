#!/bin/bash

# example:
# bash sh/submit-clm-array.sh sh/grids/clm_v0.txt

set -euo pipefail
cd ~/git/CLM

GRID_FILE=${1:?provide grid file}
GRID_BASE=$(basename "$GRID_FILE")
GRID_NAME="${GRID_BASE%.*}"

LOG_DIR="logs/$GRID_NAME"
mkdir -p "$LOG_DIR"

N=$(wc -l < "$GRID_FILE")
MAX=$((N-1))

echo "submitting array 0-$MAX from $GRID_FILE"

sbatch \
  --array=0-$MAX \
  --export=ALL,GRID_FILE="$GRID_FILE" \
  --output="$LOG_DIR/hub_%A_%a.log" \
  --error="$LOG_DIR/hub_%A_%a.log" \
  sh/run-clm-array.sh