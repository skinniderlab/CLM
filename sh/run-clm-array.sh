#!/bin/bash
#SBATCH --job-name=clm
#SBATCH --partition=main,skinniderlab
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=120:00:00
#SBATCH --output=logs/hub_%A_%a.log
#SBATCH --error=logs/hub_%A_%a.log


set -euo pipefail
cd ~/git/CLM

# -------------------------
# conda init
# -------------------------
__conda_setup="$('/Genomics/skinniderlab/ms0270/miniconda3/bin/conda' 'shell.bash' 'hook' 2>/dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    source /Genomics/skinniderlab/ms0270/miniconda3/etc/profile.d/conda.sh
fi
unset __conda_setup

conda activate clm

echo "hub job started on $(hostname)"
date


# -------------------------
# grid param selection
# -------------------------
GRID_CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$GRID_FILE")

if [ -z "$GRID_CONFIG" ]; then
    echo "no grid config file found"
    exit 1
fi

echo "array id: $SLURM_ARRAY_TASK_ID"
echo "using config: $GRID_CONFIG"


# -------------------------
# run snakemake hub
# -------------------------
snakemake all \
  --snakefile workflow/Snakefile \
  --configfile "$GRID_CONFIG" \
  --jobs 4 \
  --slurm --default-resources slurm_partition=main,skinniderlab \
  --latency-wait 60 \
  --rerun-incomplete \
  --keep-going \
  -p

echo "hub job finished"
date
