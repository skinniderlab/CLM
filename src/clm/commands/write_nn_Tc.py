import pandas as pd
from rdkit.DataStructs import FingerprintSimilarity
from clm.functions import (
    clean_mol,
    write_to_csv_file,
    compute_fingerprint,
    read_csv_file,
)
import logging


logger = logging.getLogger(__name__)


def add_args(parser):
    parser.add_argument(
        "--query_file", type=str, help="Path to the file to be queried"
    )
    parser.add_argument(
        "--reference_file",
        type=str,
        help="Path to the file that the tc is being compared to",
    )
    parser.add_argument(
        "--pubchem_file", type=str, help="Path to the PubChem file"
    )
    parser.add_argument(
        "--max_molecules",
        type=int,
        default=500_000,
        help="Number of samples to select",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the save the output file",
    )

    return parser


def calculate_fingerprint(smile):
    if (mol := clean_mol(smile, raise_error=False)) is not None:
        return compute_fingerprint(mol)
    return None


def find_max_similarity_fingerprint(
    target_smiles, target_inchikey, ref_smiles, ref_fps, ref_inchikeys
):
    target_fps = calculate_fingerprint(target_smiles)

    if target_fps is None:
        return None, None

    max_tc, max_tc_ref_smile = -1, ""
    for ref_smile, ref_fp, ref_inchikey in zip(
        ref_smiles, ref_fps, ref_inchikeys
    ):
        # Avoid comparing tcs of exactly same molecule
        if not (target_inchikey == ref_inchikey):
            if max_tc < (fps := FingerprintSimilarity(target_fps, ref_fp)):
                max_tc = fps
                max_tc_ref_smile = ref_smile

    return max_tc, max_tc_ref_smile


def prep_nn_tc(sample_file, pubchem_file, max_molecules):
    sample = read_csv_file(sample_file, delimiter=",")
    if len(sample) > max_molecules:
        sample = sample.sample(
            n=max_molecules,
            replace=True,
            weights=sample["size"],
            ignore_index=True,
        )
    pubchem = read_csv_file(pubchem_file, delimiter="\t", header=None)

    # PubChem tsv can have 3 or 4 columns (if fingerprints are precalculated)
    match len(pubchem.columns):
        case 3:
            pubchem.columns = ["smiles", "mass", "formula"]
        case 4:
            pubchem.columns = ["smiles", "mass", "formula", "fingerprint"]
            pubchem = pubchem.dropna(subset="fingerprint")
            # ignore the fingerprint column since we don't need it
            pubchem = pubchem.drop(columns="fingerprint")
        case _:
            raise RuntimeError("Unexpected column count for PubChem")

    pubchem = pubchem[pubchem["formula"].isin(set(sample.formula))]
    pubchem = pubchem.drop_duplicates(subset=["formula"], keep="first")

    combination = pd.concat(
        [
            sample.assign(source="DeepMet"),
            pubchem.assign(source="PubChem"),
        ]
    )

    return combination


def write_nn_Tc(
    query_file,
    reference_file,
    output_file,
    pubchem_file=None,
    max_molecules=None,
):
    """
    Find nearest neighbor Tanimoto coefficient for each molecule in the query file
    compared to the molecules in the reference file.
    On return, `output_file` is generated with all the rows of `query_file` and two additional columns:
    "nn_tc" and "nn" which are the nearest neighbor Tanimoto coefficient and the corresponding
    nearest neighbor molecule in the reference file respectively.
    Args:
        query_file: Query csv file containing the molecules to be compared, with
        columns "smiles" and "inchikey" (including others)
        reference_file: Reference csv file containing the molecules to be compared against,
        with columns "smiles" and "inchikey" (including others)
        output_file: Output file to save the results

    Returns:
        None
    """

    ref_fps, ref_smiles, ref_inchikeys = [], [], []

    # Processing the reference_file in chunks of one row at a time to optimize memory efficiency.
    for df in pd.read_csv(reference_file, chunksize=1, iterator=True):
        row = df.iloc[0]
        # If a particular smiles is invalid, calculate_fingerprint will return None
        if (fps := calculate_fingerprint(row.smiles)) is not None:
            ref_fps.append(fps)
            ref_smiles.append(row.smiles)
            ref_inchikeys.append(row.inchikey)

    # If PubChem file is not present, there is no preparatory computation to be done
    if pubchem_file is not None:
        query = prep_nn_tc(query_file, pubchem_file, max_molecules)
    else:
        query = read_csv_file(query_file)

    results = query.apply(
        lambda x: find_max_similarity_fingerprint(
            x.smiles,
            x.inchikey,
            ref_smiles,
            ref_fps,
            ref_inchikeys,
        ),
        axis=1,
    )
    query = query.assign(nn_tc=[i[0] for i in results])
    query = query.assign(nn=[i[1] for i in results])

    write_to_csv_file(output_file, info=query, mode="w")
    return query


def main(args):
    write_nn_Tc(
        query_file=args.query_file,
        reference_file=args.reference_file,
        output_file=args.output_file,
        pubchem_file=args.pubchem_file,
        max_molecules=args.max_molecules,
    )
