
# run batch
conda activate clm
cd workflow

SESSION=clm
INPUT_DIR="clm-input"
OUT_BASE="/scratch/tmp/sa7998/clm"

tmux has-session -t $SESSION 2>/dev/null && { echo "Session $SESSION already exists"; exit 1; }
tmux new-session -d -s $SESSION

for csv in ${INPUT_DIR}/*.csv; do
  fname=$(basename "$csv")
  name="${fname%.csv}"
  abs_csv=$(realpath "$csv")
  outdir="${OUT_BASE}/${name}"

  tmux new-window -t $SESSION -n "$name"
  tmux send-keys -t "$SESSION:$name" \
    "conda activate clm && \
    snakemake --configfile config/config_bsm.yaml \
    --config 'paths={\"output_dir\":\"$outdir\",\"dataset\":\"$abs_csv\"}' \
    --jobs 16 --slurm --latency-wait 60 --rerun-incomplete -p all" C-m
done

tmux kill-window -t $SESSION:0