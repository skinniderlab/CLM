import numpy as np
import pandas as pd
from tqdm import tqdm

from clm.functions import (
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


def add_args(parser):
    parser.add_argument("--ranks_file", type=str, help="Path to the rank file.")
    parser.add_argument(
        "--train_file", type=str, help="Path to the training dataset."
    )
    parser.add_argument(
        "--test_file", type=str, help="Path to the test dataset."
    )
    parser.add_argument(
        "--pubchem_file",
        type=str,
        help="Path to the file containing PubChem information.",
    )
    parser.add_argument(
        "--sample_file",
        type=str,
        help="Path to the file containing sample molecules.",
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
    return parser


def match_molecules(row, dataset, data_type):
    match = dataset[
        dataset["mass"].between(row["mass_range"][0], row["mass_range"][1])
    ]

    # For the PubChem dataset, not all SMILES might be valid; consider only the ones that are.
    # If a `fingerprint` column exists, then we have a valid SMILE
    if (
        len(match) > 0
        and data_type == "PubChem"
        and "fingerprint" not in dataset
    ):
        match = match[
            match.apply(
                lambda x: clean_mol(x["smiles"], raise_error=False) is not None,
                axis=1,
            )
        ]

    match = (
        match.sort_values("size", ascending=False, kind="stable")
        if data_type == "model"
        else match.sample(frac=1)
    )
    match = match.assign(rank=np.arange(match.shape[0]))
    match.columns = "target_" + match.columns

    rank = match[match["target_formula"] == row["formula"]][
        ["target_size", "target_rank", "target_source"]
    ]

    if rank.shape[0] > 1:
        rank = rank.head(1)
    elif rank.shape[0] == 0:
        rank = pd.DataFrame(
            {
                "target_size": [np.nan],
                "target_rank": [np.nan],
                "target_source": [data_type],
            }
        )
    rank = rank.assign(n_candidates=len(match.target_formula.unique()))

    rank = pd.concat(
        [
            pd.DataFrame(
                [
                    row[
                        [
                            "smiles",
                            "mass",
                            "formula",
                            "mass_known",
                            "formula_known",
                        ]
                    ]
                ]
            ).reset_index(drop=True),
            rank.reset_index(drop=True),
        ],
        axis=1,
    )

    return rank


def write_formula_prior_CV(
    ranks_file,
    train_file,
    test_file,
    pubchem_file,
    sample_file,
    err_ppm,
    chunk_size,
):
    train = generate_df(train_file, chunk_size)
    train = train.assign(size=np.nan)

    test = generate_df(test_file, chunk_size)
    test["mass_range"] = test.apply(
        lambda x: get_mass_range(x["mass"], err_ppm), axis=1
    )
    test = test.assign(mass_known=test["mass"].isin(train["mass"]))
    test = test.assign(formula_known=test["formula"].isin(train["formula"]))

    print("Reading PubChem file")
    pubchem = read_csv_file(pubchem_file, delimiter="\t", header=None)

    # PubChem tsv can have 3, 4 or 5 columns
    match len(pubchem.columns):
        case 3:
            pubchem.columns = ["smiles", "mass", "formula"]
        case 4:
            pubchem.columns = ["smiles", "mass", "formula", "fingerprint"]
            pubchem = pubchem.dropna(subset="fingerprint")
        case 5:
            pubchem.columns = [
                "smiles",
                "mass",
                "formula",
                "fingerprint",
                "inchikey",
            ]
            pubchem = pubchem.dropna(subset="fingerprint")
        case _:
            raise RuntimeError("Unexpected column count for PubChem")

    pubchem = pubchem.assign(size=np.nan)

    print("Reading sample file from generative model")
    gen = read_csv_file(sample_file)

    # iterate through PubChem vs. generative model
    inputs = {
        "model": gen.assign(source="model"),
        "PubChem": pubchem.assign(source="PubChem"),
        "train": train.assign(source="train"),
    }

    rank_df = pd.DataFrame()
    for datatype, dataset in inputs.items():
        print(f"Generating statistics for model {datatype}")

        result = test.progress_apply(
            lambda x: match_molecules(x, dataset, datatype), axis=1
        )

        rank = pd.concat(result.to_list())
        rank.insert(0, "Index", range(len(rank)))
        rank_df = pd.concat([rank_df, rank])

    write_to_csv_file(ranks_file, rank_df)


def main(args):
    write_formula_prior_CV(
        ranks_file=args.ranks_file,
        train_file=args.train_file,
        test_file=args.test_file,
        pubchem_file=args.pubchem_file,
        sample_file=args.sample_file,
        err_ppm=args.err_ppm,
        chunk_size=args.chunk_size,
    )
