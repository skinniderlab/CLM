# check size per folder
du -h .snakemake/slurm_logs/* | sort -h

# remove logs (-mtime +days)
find .snakemake/slurm_logs -type f -name "*.log" -mtime +30 -delete