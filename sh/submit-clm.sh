

conda activate clm

cd workflow
snakemake --configfile config/config_fast.yaml --jobs 4 --slurm --latency-wait 60 --rerun-incomplete

# run
cd python/CLM/workflow
snakemake --configfile config/config_bsm.yaml --jobs 4 --dry-run -p all
snakemake --configfile config/config_bsm.yaml --jobs 4 --slurm --latency-wait 60 --rerun-incomplete -p all


##

snakemake --configfile config/config.yaml --jobs 8 --slurm
snakemake --configfile config/config_fast.yaml --jobs 8 --slurm --latency-wait 60 --rerun-incomplete
snakemake -s Snakefile-mini --configfile config/config_fast.yaml --jobs 8 --slurm --latency-wait 60 --rerun-incomplete
snakemake -s Snakefile --configfile config/config_fast.yaml --jobs 8 --slurm --latency-wait 60 --rerun-incomplete

  # The input dataset file.
  
  python/CLM/workflow/config/config.yaml



snakemake --configfile config/config.yaml --jobs 8 --slurm \
  | tee snakemake.log

cat python/CLM/workflow/.snakemake/log/2025-12-22T142215.163725.snakemake.log
cat python/CLM/workflow/.snakemake/slurm_logs/rule_sample_molecules_RNN/3221584.log
cat python/CLM/workflow/.snakemake/log/2025-12-23T023720.140407.snakemake.log
cat python/CLM/workflow/.snakemake/slurm_logs/rule_train_models_RNN/3227493.log

cat python/CLM/workflow/.snakemake/log/2025-12-23T172157.581892.snakemake.log

cat python/CLM/workflow/.snakemake/log/2025-12-23T172157.581892.snakemake.log

cat python/CLM/workflow/.snakemake/log/2025-12-23T155330.089928.snakemake.log

cat python/CLM/workflow/.snakemake/slurm_logs/rule_preprocess/3691979.log


ls -lh python/CLM/workflow/data/0/prior/samples/LOTUS_truncated_SMILES_1_unique_masses.csv.gz

rm -r /home/sa7998/git/biosphere-metabolome/python/CLM/workflow/data

cat python/CLM/workflow/config/config_fast.yaml


ls python/CLM/workflow
ls python/CLM/workflow/data