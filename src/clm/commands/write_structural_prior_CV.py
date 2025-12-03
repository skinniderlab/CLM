import numpy as np
import logging
import pandas as pd
from rdkit import Chem
from rdkit.DataStructs import FingerprintSimilarity, ExplicitBitVect
from tqdm import tqdm

from clm.functions import (
    compute_fingerprint,
    compute_fingerprints,
    clean_mol,
    generate_df,
    get_mass_range,
    write_to_csv_file,
    read_csv_file,
)

# suppress rdkit errors
from rdkit import rdBase

rdBase.DisableLog("rdApp.error")
tqdm.pandas()
logger = logging.getLogger(__name__)


def add_args(parser):
    parser.add_argument("--ranks_file", type=str, help="Path to the rank file.")
    parser.add_argument("--tc_file", type=str, help="Path to the tc file.")
    parser.add_argument("--train_file", type=str, help="Path to the training dataset.")
    parser.add_argument("--test_file", type=str, help="Path to the test dataset.")
    parser.add_argument(
        "--pubchem_file",
        type=str,
        help="Path to the file containing PubChem information.",
    )
    parser.add_argument(
        "--carbon_file",
        type=str,
        default=None,
        help="Path to the training dataset altered by the add carbon step.",
    )
    parser.add_argument(
        "--cv_ranks_files",
        type=str,
        nargs="+",
        help="Rank files for individual CV folds.",
    )
    parser.add_argument(
        "--cv_tc_files",
        type=str,
        nargs="+",
        help="Tc files for individual CV folds.",
    )
    parser.add_argument(
        "--sample_file", type=str, help="Path to the file containing sample molecules."
    )
    parser.add_argument(
        "--err_ppm",
        type=int,
        help="Error margin in parts per million for chemical analysis.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100000,
        help="Size of chunks for processing large files.",
    )
    parser.add_argument(
        "--top_n", type=int, default=1, help="Max. number of top ranks to save for Tc."
    )
    return parser


def get_fp_obj(fp_string, bits=1024):
    # Get FingerPrint object from a Base64 fingerprint string
    # There currently seems to be no more concise way to do this since
    # .FromBase64 doesn't return the modified object
    fp = ExplicitBitVect(bits)
    fp.FromBase64(fp_string)
    return fp


def get_inchikey(smile):
    # Get Inchikey for a valid smile
    return Chem.inchi.MolToInchiKey(clean_mol(smile, raise_error=True))


def match_molecules(row, dataset, data_type, top_n=1):
    match = dataset[dataset["mass"].between(row["mass_range"][0], row["mass_range"][1])]

    # For the PubChem dataset, not all SMILES might be valid; consider only the ones that are.
    # If a `fingerprint` column exists, then we have a valid SMILE
    if len(match) > 0 and data_type == "PubChem" and "fingerprint" not in dataset:
        match = match[
            match.apply(
                lambda x: clean_mol(x["smiles"], raise_error=False) is not None, axis=1
            )
        ]

    # Assign sequential ranks based on decreasing frequency of sampling
    match = (
        match.sort_values("size", ascending=False, kind="stable")
        if data_type == "model"
        else match.sample(frac=1)
    )
    match = match.assign(rank=np.arange(match.shape[0]))
    # We'll be adding columns to `match` later. For now keep track of
    # what the `target_<rank/smiles/mass/formula>` values are for the given
    # source (model/train/PubChem)
    match.columns = "target_" + match.columns

    # For Pubchem, Inchi keys may not have been pre-calculated
    if data_type == "PubChem" and "target_inchikey" not in match.columns:
        match["target_inchikey"] = match["target_smiles"].apply(
            lambda x: get_inchikey(x)
        )

    row_inchikey = row["inchikey"]
    rank = match[match["target_inchikey"] == row_inchikey][
        ["target_size", "target_rank", "target_source", "target_inchikey"]
    ]

    # `rank` denotes the best match
    if rank.shape[0] > 1:
        rank = rank.head(1)
    elif rank.shape[0] == 0:
        rank = pd.DataFrame(
            {
                "target_size": [np.nan],
                "target_rank": [np.nan],
                "target_source": [data_type],
                "target_inchikey": [row_inchikey],
            }
        )
    # `n_candidates` is the number of candidates close enough to the NPS
    # in terms of molecular mass.
    rank = rank.assign(n_candidates=match.shape[0])

    tc = match
    if tc.shape[0] > top_n:
        tc = tc.head(top_n)

    if tc.shape[0] > 0:
        if "target_fingerprint" in tc:
            target_fps = [get_fp_obj(fp) for fp in tc["target_fingerprint"]]
        else:
            target_mols = [
                clean_mol(smile, selfies=False) for smile in tc["target_smiles"].values
            ]
            target_fps = compute_fingerprints(target_mols, algorithm="ecfp6")
        tc["Tc"] = [
            FingerprintSimilarity(row["fp"], target_fp) for target_fp in target_fps
        ]
    else:
        empty_data = {key: np.nan for key in tc.columns}
        minimal_data = {"target_source": data_type, "target_inchikey": row_inchikey}
        tc = pd.DataFrame(
            empty_data | minimal_data,
            index=[0],
        )

    row = row[["smiles", "mass", "formula", "mass_known", "formula_known"]]
    tc = pd.concat(
        [
            pd.DataFrame(
                np.repeat(pd.DataFrame([row]), repeats=len(tc), axis=0),
                columns=["smiles", "mass", "formula", "mass_known", "formula_known"],
            ).reset_index(drop=True),
            tc.reset_index(drop=True),
        ],
        axis=1,
    )

    rank = pd.concat(
        [
            pd.DataFrame([row]).reset_index(drop=True),
            rank.reset_index(drop=True),
        ],
        axis=1,
    )

    return pd.Series((rank, tc))


def write_structural_prior_CV(
    ranks_file,
    tc_file,
    train_file,
    test_file,
    pubchem_file,
    sample_file,
    err_ppm,
    chunk_size,
    carbon_file=None,
    cv_ranks_files=None,
    cv_tc_flies=None,
    top_n=1,
):
    train = generate_df(train_file, chunk_size)
    train = train.assign(size=np.nan)

    test = generate_df(test_file, chunk_size)
    test["fp"] = test.apply(
        lambda row: compute_fingerprint(clean_mol(row["smiles"]), algorithm="ecfp6"),
        axis=1,
    )
    test["mass_range"] = test.apply(
        lambda x: get_mass_range(x["mass"], err_ppm), axis=1
    )
    test = test.assign(mass_known=test["mass"].isin(train["mass"]))
    test = test.assign(formula_known=test["formula"].isin(train["formula"]))

    logger.info("Reading sample file from generative model")
    gen = read_csv_file(sample_file)

    inputs = {
        "model": gen.assign(source="model")
    }

    if pubchem_file:
        logger.info("Reading PubChem file")
        pubchem = read_csv_file(pubchem_file, delimiter="\t", header=None)

        # PubChem tsv can have 3, 4 or 5 columns
        match len(pubchem.columns):
            case 3:
                pubchem.columns = ["smiles", "mass", "formula"]
            case 4:
                pubchem.columns = ["smiles", "mass", "formula", "fingerprint"]
                pubchem = pubchem.dropna(subset="fingerprint")
            case 5:
                pubchem.columns = ["smiles", "mass", "formula", "fingerprint", "inchikey"]
                pubchem = pubchem.dropna(subset="fingerprint")
            case _:
                raise RuntimeError("Unexpected column count for PubChem")

        pubchem = pubchem.assign(size=np.nan)
        inputs['PubChem'] = pubchem.assign(source="PubChem")

    # We are only comparing training set with test set for individual cv fold
    if cv_ranks_files is None and cv_tc_flies is None:
        inputs["train"] = train.assign(source="train")

    if carbon_file:
        addcarbon = read_csv_file(carbon_file)

        # Rename carbon's column names to coincide with other inputs and drop input smiles
        addcarbon.rename(columns={"mutated_smiles": "smiles"}, inplace=True)
        addcarbon.drop(columns="input_smiles", inplace=True)

        # assign frequencies of add carbons as none
        addcarbon = addcarbon.assign(size=np.nan)

        inputs["addcarbon"] = addcarbon.assign(source="addcarbon")

    rank_df, tc_df = pd.DataFrame(), pd.DataFrame()
    for datatype, dataset in inputs.items():
        logging.info(f"Generating statistics for model {datatype}")

        results = test.progress_apply(
            lambda x: match_molecules(x, dataset, datatype, top_n=top_n), axis=1
        )

        logging.info(f"Generated statistics for model {datatype}")
        rank = pd.concat(results[0].to_list(), axis=0)
        rank.insert(0, "Index", range(len(rank)))
        tc = pd.concat(results[1].to_list(), axis=0)
        tc.insert(0, "Index", range(len(tc)))
        rank_df = pd.concat([rank_df, rank], axis=0)
        tc_df = pd.concat([tc_df, tc], axis=0)

    # The cv_rank_files contain statistics evaluated from individual cross-validation folds
    # Since the test sets across all folds and the train set across all fold contain exactly the same unique elements,
    # we aggregate results from all folds to assess test SMILES against training SMILES across all folds
    if cv_ranks_files is not None and cv_tc_flies is not None:
        cv_rank_data = pd.concat([read_csv_file(f) for f in cv_ranks_files], axis=0)
        cv_tc_data = pd.concat([read_csv_file(f) for f in cv_tc_flies], axis=0)

        train_cv_rank_data = cv_rank_data[cv_rank_data["target_source"] == "train"]
        train_cv_tc_data = cv_tc_data[cv_tc_data["target_source"] == "train"]

        rank_df = pd.concat([rank_df, train_cv_rank_data], axis=0)
        tc_df = pd.concat([tc_df, train_cv_tc_data], axis=0)

    write_to_csv_file(ranks_file, rank_df)
    write_to_csv_file(
        tc_file, tc_df.drop("target_fingerprint", axis=1, errors="ignore")
    )


def main(args):
    write_structural_prior_CV(
        ranks_file=args.ranks_file,
        tc_file=args.tc_file,
        train_file=args.train_file,
        test_file=args.test_file,
        pubchem_file=args.pubchem_file,
        sample_file=args.sample_file,
        err_ppm=args.err_ppm,
        chunk_size=args.chunk_size,
        carbon_file=args.carbon_file,
        cv_ranks_files=args.cv_ranks_files,
        cv_tc_flies=args.cv_tc_files,
        top_n=args.top_n,
    )
