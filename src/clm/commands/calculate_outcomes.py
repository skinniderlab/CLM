import os
import numpy as np
import pandas as pd
import scipy.stats
from fcd_torch import FCD
import logging
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem.GraphDescriptors import BertzCT
from rdkit.Chem.MolSurf import TPSA
from rdkit.Chem.QED import qed
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from rdkit import rdBase
from rdkit.Contrib.SA_Score import sascorer
from rdkit.Contrib.NP_Score import npscorer
from tqdm import tqdm

from clm.functions import (
    clean_mol,
    write_to_csv_file,
    compute_fingerprint,
    split_frequency_ranges,
    # Functions for calculating metrics
    continuous_JSD,
    discrete_JSD,
    internal_diversity,
    external_diversity,
    internal_nn,
    external_nn,
    pct_rotatable_bonds,
    pct_stereocenters,
    read_csv_file,
)

rdBase.DisableLog("rdApp.error")
fscore = npscorer.readNPModel()
logger = logging.getLogger(__name__)
tqdm.pandas()


def add_args(parser):
    parser.add_argument(
        "--train_file",
        type=str,
        help="Training csv file with smiles as a column.",
    )
    parser.add_argument(
        "--sampled_file", type=str, help="Path to the sampled file"
    )
    parser.add_argument(
        "--invalid_smiles_file",
        type=str,
        help="Path to the invalid sampled file",
    )
    parser.add_argument(
        "--known_smiles_file", type=str, help="Path to the known sampled file"
    )
    parser.add_argument(
        "--max_molecules",
        type=int,
        help="Max number of sampled smiles to select for a bin (smiles that are not designated to a bin are discarded.)",
        default=None,
    )
    parser.add_argument("--output_file", type=str)
    return parser


def safe_qed(mol):
    try:
        return qed(mol)
    except OverflowError:
        return None


def safe_sascorer(mol):
    try:
        return sascorer.calculateScore(mol)
    except (OverflowError, ZeroDivisionError):
        return None


def murcko_scaffold_smiles(mol):
    try:
        return MurckoScaffoldSmiles(mol=mol)
    except Chem.rdchem.AtomValenceException:
        return None


molecular_properties = {
    "canonical_smile": Chem.MolToSmiles,
    "elements": lambda mol: [atom.GetSymbol() for atom in mol.GetAtoms()],
    "mws": Descriptors.MolWt,
    "logp": Descriptors.MolLogP,
    "tcs": BertzCT,
    "tpsa": TPSA,
    "qed": safe_qed,
    "rings1": Lipinski.RingCount,
    "rings2": Lipinski.NumAliphaticRings,
    "rings3": Lipinski.NumAromaticRings,
    "SA": safe_sascorer,
    "NP": lambda mol: npscorer.scoreMol(mol, fscore),
    "sp3": Lipinski.FractionCSP3,
    "rot": pct_rotatable_bonds,
    "stereo": pct_stereocenters,
    "murcko": murcko_scaffold_smiles,
    "donors": Lipinski.NumHDonors,
    "acceptors": Lipinski.NumHAcceptors,
    "fps": compute_fingerprint,
}


def smile_properties_dataframe(a_row, is_sample=False):
    data = []

    row = [None]
    if (mol := clean_mol(a_row.smiles, raise_error=False)) is not None:
        row = tuple(fun(mol) for fun in molecular_properties.values())

    if None in row:
        row = tuple([None] * len(molecular_properties))

    if is_sample:
        # note: "size" instead of .size which is a property of Series
        data.append(
            (
                a_row.smiles,
                a_row.is_valid,
                a_row.is_novel,
                a_row["size"],
                a_row.bin,
            )
            + row
        )
    else:
        data.append((a_row.smiles,) + row)

    # Unlike training smiles, sampled smiles can be categorized as valid/ novel
    # So, sampled df have more columns than train df, both of which are parsed in this function
    columns = (
        ["smiles", "is_valid", "is_novel", "size", "bin"]
        + list(molecular_properties.keys())
        if is_sample
        else ["smiles"] + list(molecular_properties.keys())
    )

    # Some of the calculations later strictly require specific datatypes
    # The presence of nan values messes up dtypes of some columns
    # Using convert_dtypes to convert columns to the best possible dtypes
    df = pd.DataFrame(data, columns=columns).convert_dtypes()
    return df


def calculate_probabilities(*dicts):
    merged_keys = set()
    for dictionary in dicts:
        merged_keys.update(dictionary.keys())

    return_values = []
    for dictionary in dicts:
        total = sum(dictionary.values())
        return_values.append(
            [dictionary.get(k, 0) / total for k in merged_keys]
        )

    return return_values


def get_dataframes(train_file, prep_sample_df):
    logger.info(f"Reading training smiles from {train_file}")

    train_data = []
    for df in read_csv_file(train_file, chunksize=1000):
        train_data.extend(df.apply(smile_properties_dataframe, axis=1))

    train_df = pd.concat(train_data)

    logger.info(f"Reading sample smiles from {prep_sample_df}")
    # Compute molecular properties of valid SMILES only
    valid_data = prep_sample_df[prep_sample_df["is_valid"]].progress_apply(
        lambda x: smile_properties_dataframe(x, is_sample=True), axis=1
    )
    if valid_data.empty:
        valid_df = prep_sample_df.iloc[:0]  # empty
    else:
        valid_df = pd.concat(valid_data.to_list())

    if len(valid_data) == len(prep_sample_df):
        invalid_df = prep_sample_df.iloc[:0]  # empty
    else:
        invalid_df = prep_sample_df[~(prep_sample_df["is_valid"])]

    # Concatenate valid and invalid dfs for computations later
    sample_df = pd.concat([valid_df, invalid_df])

    n_valid_smiles = sample_df["is_valid"].sum()
    logger.info(f"{n_valid_smiles} valid SMILES out of {len(sample_df)}")

    n_novel_smiles = sample_df["is_novel"].sum()
    logger.info(f"{n_novel_smiles} novel SMILES out of {len(sample_df)}")

    return train_df, sample_df


def calculate_outcomes_dataframe(sample_df, train_df):
    def handle_bin(
        bin_name, df, train_element_distribution, train_murcko_distribution
    ):
        n_total_smiles = df["size"].sum()

        # Filtering out invalid smiles
        bin_df = df[df["is_valid"]]
        if "canonical_smile" in df.columns:
            bin_df = df[~df["canonical_smile"].isnull()]

        n_valid_smiles = bin_df["size"].sum()

        # Skip iteration if number of valid smiles in a particular bin is 0
        if len(bin_df) == 0:
            return None

        bin_df = bin_df.reset_index(drop=True)
        element_distribution = dict(
            zip(
                *np.unique(
                    np.concatenate(bin_df["elements"].to_numpy()),
                    return_counts=True,
                )
            )
        )

        murcko_distribution = dict(
            zip(*np.unique(bin_df["murcko"].to_numpy(), return_counts=True))
        )

        p1, p2 = calculate_probabilities(
            train_element_distribution, element_distribution
        )
        p_m1, p_m2 = calculate_probabilities(
            train_murcko_distribution, murcko_distribution
        )
        fcd = FCD(canonize=False)

        return {
            "bin": bin_name,
            "n_mols": n_total_smiles,
            "% valid": n_valid_smiles / n_total_smiles,
            "% novel": bin_df[bin_df["is_novel"]]["size"].sum()
            / n_valid_smiles,
            "% unique": len(bin_df) / n_valid_smiles,
            "KL divergence, atoms": scipy.stats.entropy(p2, p1),
            "Jensen-Shannon distance, atoms": jensenshannon(p2, p1),
            "Wasserstein distance, atoms": wasserstein_distance(p2, p1),
            "Jensen-Shannon distance, MWs": continuous_JSD(
                bin_df["mws"], train_df["mws"]
            ),
            "Jensen-Shannon distance, logP": continuous_JSD(
                bin_df["logp"], train_df["logp"]
            ),
            "Jensen-Shannon distance, Bertz TC": continuous_JSD(
                bin_df["tcs"], train_df["tcs"]
            ),
            "Jensen-Shannon distance, QED": continuous_JSD(
                bin_df["qed"], train_df["qed"]
            ),
            "Jensen-Shannon distance, TPSA": continuous_JSD(
                bin_df["tpsa"], train_df["tpsa"]
            ),
            "Internal diversity": internal_diversity(bin_df["fps"].to_numpy()),
            "External diversity": external_diversity(
                bin_df["fps"].to_numpy(), train_df["fps"].to_numpy()
            ),
            "Internal nearest-neighbor Tc": internal_nn(
                bin_df["fps"].to_numpy()
            ),
            "External nearest-neighbor Tc": external_nn(
                bin_df["fps"].to_numpy(), train_df["fps"].to_numpy()
            ),
            "Jensen-Shannon distance, # of rings": discrete_JSD(
                bin_df["rings1"], train_df["rings1"]
            ),
            "Jensen-Shannon distance, # of aliphatic rings": discrete_JSD(
                bin_df["rings2"], train_df["rings2"]
            ),
            "Jensen-Shannon distance, # of aromatic rings": discrete_JSD(
                bin_df["rings3"], train_df["rings3"]
            ),
            "Jensen-Shannon distance, SA score": continuous_JSD(
                bin_df["SA"], train_df["SA"]
            ),
            "Jensen-Shannon distance, NP score": continuous_JSD(
                bin_df["NP"], train_df["NP"]
            ),
            "Jensen-Shannon distance, % sp3 carbons": continuous_JSD(
                bin_df["sp3"], train_df["sp3"]
            ),
            "Jensen-Shannon distance, % rotatable bonds": continuous_JSD(
                bin_df["rot"], train_df["rot"]
            ),
            "Jensen-Shannon distance, % stereocenters": continuous_JSD(
                bin_df["stereo"], train_df["stereo"]
            ),
            "Jensen-Shannon distance, Murcko scaffolds": jensenshannon(
                p_m2, p_m1
            ),
            "Jensen-Shannon distance, hydrogen donors": discrete_JSD(
                bin_df["donors"], train_df["donors"]
            ),
            "Jensen-Shannon distance, hydrogen acceptors": discrete_JSD(
                bin_df["acceptors"], train_df["acceptors"]
            ),
            "Frechet ChemNet distance": fcd(
                bin_df[bin_df["is_novel"]]["canonical_smile"].to_numpy(),
                train_df["canonical_smile"].to_numpy(),
            ),
        }

    train_element_distribution = dict(
        zip(
            *np.unique(
                np.concatenate(train_df["elements"].to_numpy()),
                return_counts=True,
            )
        )
    )
    train_murcko_distribution = dict(
        zip(*np.unique(train_df["murcko"].to_numpy(), return_counts=True))
    )

    out = []
    for bin, bin_df in sample_df.groupby("bin"):
        logger.info(f"Calculating outcomes for bin {bin}")
        if _out := handle_bin(
            bin, bin_df, train_element_distribution, train_murcko_distribution
        ):
            out.append(_out)

    out = pd.DataFrame(out)

    # Have 'bin' as a column and each of our other columns as rows in an
    # 'outcome' column, with values in a 'value' column
    if not out.empty:
        out = out.melt(id_vars=["bin"], var_name="outcome", value_name="value")
    return out


def prep_outcomes_freq(
    samples,
    max_molecules,
    known_smiles,
    invalid_smiles,
):
    known_df = read_csv_file(known_smiles, usecols=["smiles", "size"]).assign(
        is_valid=True, is_novel=False
    )
    invalid_df = read_csv_file(
        invalid_smiles, usecols=["smiles", "size"]
    ).assign(is_valid=False, is_novel=True)
    sample_df = read_csv_file(samples, usecols=["smiles", "size"]).assign(
        is_valid=True, is_novel=True
    )

    data = pd.concat([known_df, invalid_df, sample_df])
    data = split_frequency_ranges(data, max_molecules, all=True)

    return data


def calculate_outcomes(
    sampled_file,
    train_file,
    known_smiles,
    invalid_smiles,
    max_molecules,
    output_file,
):
    prep_sample_df = prep_outcomes_freq(
        sampled_file, max_molecules, known_smiles, invalid_smiles
    )
    train_df, sample_df = get_dataframes(train_file, prep_sample_df)

    logger.info("Calculating outcomes")
    out = calculate_outcomes_dataframe(sample_df, train_df)

    # `input_file` column added for legacy reasons
    out["input_file"] = os.path.basename(sampled_file)

    write_to_csv_file(output_file, out)
    return out


def main(args):
    calculate_outcomes(
        train_file=args.train_file,
        sampled_file=args.sampled_file,
        known_smiles=args.known_smiles_file,
        invalid_smiles=args.invalid_smiles_file,
        max_molecules=args.max_molecules,
        output_file=args.output_file,
    )
