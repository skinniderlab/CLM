find workflow/.snakemake/slurm_logs -type f -exec dirname {} \; | sort | uniq -c

LOGDIR=workflow/.snakemake/slurm_logs/rule_train_models_RNN
LOGDIR=workflow/.snakemake/slurm_logs/rule_sample_molecules_RNN
LOGDIR=workflow/.snakemake/slurm_logs/rule_tabulate_molecules
LOGDIR=workflow/.snakemake/slurm_logs/rule_collect_tabulated_molecules


qos_sacct workflow/.snakemake/slurm_logs/rule_train_models_RNN
qos_sacct workflow/.snakemake/slurm_logs/rule_sample_molecules_RNN
qos_sacct workflow/.snakemake/slurm_logs/rule_tabulate_molecules
qos_sacct workflow/.snakemake/slurm_logs/rule_collect_tabulated_molecules
qos_sacct workflow/.snakemake/slurm_logs/rule_process_tabulated_molecules

jobids=$(ls $LOGDIR/*.log | sed 's#.*/##; s/\.log//' | paste -sd, -)

sacct -j $jobids \
  --format=JobID,JobName,State,Elapsed,Timelimit,AllocCPUS,TotalCPU,MaxRSS,ReqMem \
  | awk '$1 !~ /\./'

# including child jobs
sacct -j $jobids \
  --format=JobID,JobName,State,Elapsed,Timelimit,AllocCPUS,TotalCPU,MaxRSS,ReqMem,AllocTRES



###

qos_sacct () {
  if [ -z "$1" ]; then
    echo "Usage: qos_sacct <logdir>"
    return 1
  fi

  local LOGDIR=$1
  echo $LOGDIR
  local jobids=$(ls "$LOGDIR"/*.log 2>/dev/null | sed 's#.*/##; s/\.log//' | paste -sd, -)

  if [ -z "$jobids" ]; then
    echo "No log files found in $LOGDIR."
    return 1
  fi

  sacct -j "$jobids" \
    --format=JobID,JobName,State,Elapsed,Timelimit,AllocCPUS,TotalCPU,MaxRSS,ReqMem \
  | awk '$1 !~ /\./'
}