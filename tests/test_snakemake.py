from pathlib import Path
import snakemake
import tempfile
import hashlib
import logging
import gzip
import clm  # noqa: F401 (imported to enable logging configuration)


base_dir = Path(__file__).parent.parent

config_file = base_dir / "workflow/config/config_fast.yaml"
dataset = base_dir / "tests/test_data/LOTUS_truncated.txt"
pubchem_tsv_file = base_dir / "tests/test_data/PubChem_truncated.tsv"

logger = logging.getLogger("clm")


def test_snakemake():
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Running snakemake workflow in {temp_dir}")
    success = snakemake.snakemake(
        snakefile=str(base_dir / "workflow/Snakefile"),
        workdir=str(base_dir / "workflow"),
        cores=1,
        configfiles=[config_file],
        config={
            "random_seed": 5831,
            "paths": {
                "dataset": dataset,
                "pubchem_tsv_file": pubchem_tsv_file,
                "output_dir": temp_dir,
            },
            "representations": ["SMILES"],
            "enum_factors": [0],
            "folds": 3,
            "sample_seeds": [0],
            "structural_prior_min_freq": [1],
        },
        dryrun=False,
        latency_wait=60,
        forceall=True,
        verbose=True,
        printshellcmds=True,
        use_conda=True,
    )
    assert success, "Workflow did not complete successfully"

    ranks_file_overall = f"{temp_dir}/0/prior/structural_prior/LOTUS_truncated_SMILES_min1_all_freq-avg_CV_ranks_structure.csv.gz"
    checksum = hashlib.md5(
        "".join(
            [
                line.decode("utf8")
                for line in gzip.open(ranks_file_overall, "r").readlines()
            ]
        ).encode("utf8")
    ).hexdigest()
    assert checksum == "8574598a5961eed1d2e48341d343a997"
    #
    tc_file_overall = f"{temp_dir}/0/prior/structural_prior/LOTUS_truncated_SMILES_min1_all_freq-avg_CV_tc.csv.gz"
    checksum = hashlib.md5(
        "".join(
            [
                line.decode("utf8")
                for line in gzip.open(tc_file_overall, "r").readlines()
            ]
        ).encode("utf8")
    ).hexdigest()
    assert checksum == "9347406876ec6853885689f4dafd142b"
