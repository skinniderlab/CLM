"""
Calculate fingerprints on smiles in an input tsv file and write them out to a
new tsv file
"""

import os
import sys
import logging
import numpy as np
from rdkit import rdBase
from concurrent.futures import ThreadPoolExecutor, as_completed
from clm.functions import clean_mol, compute_fingerprint, read_csv_file


rdBase.DisableLog("rdApp.error")
logger = logging.getLogger("clm")


def process_chunk(df, indices):
    def fp(row):
        mol = clean_mol(row["smile"], selfies=False, raise_error=False)
        fp = (
            compute_fingerprint(mol, algorithm="ecfp6").ToBase64()
            if mol is not None
            else ""
        )
        return fp

    return df.iloc[indices].apply(lambda row: fp(row), axis=1)


if __name__ == "__main__":

    if len(sys.argv) < 3:
        logger.info("Usage: <this_script> <input_tsv> <output_tsv>")
        sys.exit(0)

    input_tsv, output_tsv = sys.argv[1:]

    logger.info("Reading PubChem file")
    df = read_csv_file(
        input_tsv,
        delimiter="\t",
        header=None,
        names=["smile", "mass", "formula"],
    )

    n_threads = int(os.environ.get("SLURM_NPROCS", 1))
    logger.log(
        msg=f"Using {n_threads} threads",
        level=logging.INFO if n_threads > 1 else logging.WARNING,
    )

    chunks_indices = np.array_split(np.arange(len(df)), n_threads)
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        future_map = {}
        for i, chunk_indices in enumerate(chunks_indices):
            future = executor.submit(process_chunk, df, chunk_indices)
            future_map[future] = i

    for future in as_completed(future_map):
        i = future_map[future]
        values = future.result()
        df.loc[values.index, "fp"] = values

    df.to_csv(output_tsv, sep="\t", header=False, index=False)
