#!/bin/bash

# example:
# bash sh/submit-clm-array.sh sh/grids/grid_bsm_AL.txt

set -euo pipefail
cd ~/git/CLM
mkdir -p logs

GRID_FILE=${1:?provide grid file}


N=$(wc -l < "$GRID_FILE")
MAX=$((N-1))

echo "submitting array 0-$MAX from $GRID_FILE"

sbatch \
  --array=0-$MAX \
  --export=ALL,GRID_FILE="$GRID_FILE" \
  sh/run-clm-array.sh