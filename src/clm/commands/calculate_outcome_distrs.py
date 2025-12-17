"""
Calculate a set of outcomes for a list of SMILES, writing the complete
distribution and not just a summary statistic.
"""

import os
import numpy as np
import pandas as pd
from itertools import chain
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
from rdkit.Chem.GraphDescriptors import BertzCT
from rdkit.Contrib.SA_Score import sascorer
from rdkit.Contrib.NP_Score import npscorer
from rdkit.Chem.rdMolDescriptors import (
    CalcNumAmideBonds,
    CalcNumHeterocycles,
    CalcNumBridgeheadAtoms,
    CalcNumSpiroAtoms,
)
from tqdm import tqdm

# import functions
from clm.functions import (
    read_file,
    clean_mol,
    pct_rotatable_bonds,
    pct_stereocenters,
    write_to_csv_file,
    read_csv_file,
)

# suppress Chem.MolFromSmiles error output
from rdkit import rdBase

rdBase.DisableLog("rdApp.error")


def add_args(parser):
    parser.add_argument(
        "--sample_file", type=str, help="Path to the sampled file"
    )
    parser.add_argument("--train_file", type=str, help="Path to the train file")
    parser.add_argument(
        "--max_mols", type=int, help="Number of samples to select"
    )
    parser.add_argument(
        "--pubchem_file", type=str, help="Path to the PubChem file"
    )
    parser.add_argument(
        "--output_file", type=str, help="Path to the save the output file"
    )
    return parser


def write_outcome_distr(sample_file, max_mols, train_file, pubchem_file):
    sample = read_csv_file(sample_file, delimiter=",")
    if sample.shape[0] > max_mols:
        sample = sample.sample(
            n=max_mols, replace=True, weights=sample["size"], ignore_index=True
        )

    pubchem = read_csv_file(
        pubchem_file,
        delimiter="\t",
        header=None,
        names=["smiles", "mass", "formula"],
    )
    pubchem = pubchem[pubchem["formula"].isin(set(sample.formula))]
    pubchem = pubchem.drop_duplicates(subset=["formula"], keep="first")

    train = pd.DataFrame(
        {"smiles": read_file(train_file, smile_only=True)["smiles"]}
    )
    combination = pd.concat(
        [
            sample.assign(source="model"),
            pubchem.assign(source="pubchem"),
            train.assign(source="train"),
        ]
    )

    # write_to_csv_file(output_file, combination, columns=["smiles", "source"])
    return combination[["smiles", "source"]]


def calculate_outcome_distr(
    sample_file, max_mols, train_file, pubchem_file, output_file
):
    # create results container
    res = []

    # read SMILES and convert to molecules
    df = write_outcome_distr(sample_file, max_mols, train_file, pubchem_file)
    smiles = df["smiles"].tolist()
    mols = [clean_mol(smile, raise_error=False) for smile in smiles]
    idxs = [idx for idx, mol in enumerate(mols) if mol]
    mols = [mols[idx] for idx in idxs]
    smiles = [smiles[idx] for idx in idxs]
    df = df.iloc[idxs, :]

    # calculate descriptors
    # heteroatom distribution
    elements = [[atom.GetSymbol() for atom in mol.GetAtoms()] for mol in mols]
    counts = np.unique(list(chain(*elements)), return_counts=True)
    # molecular weights
    mws = [Descriptors.MolWt(mol) for mol in mols]
    # logP
    logp = [Descriptors.MolLogP(mol) for mol in tqdm(mols)]
    # Bertz TC
    tcs = [BertzCT(mol) for mol in tqdm(mols)]
    # TPSA
    tpsa = [Descriptors.TPSA(mol) for mol in mols]
    # QED
    qed = []
    for mol in tqdm(mols):
        try:
            qed.append(Descriptors.qed(mol))
        except OverflowError:
            pass

    # % of sp3 carbons
    pct_sp3 = [Lipinski.FractionCSP3(mol) for mol in tqdm(mols)]
    # % rotatable bonds
    pct_rot = [pct_rotatable_bonds(mol) for mol in mols]
    # % of stereocentres
    pct_stereo = [pct_stereocenters(mol) for mol in mols]
    # % heteroatoms
    pct_hetero = [
        Lipinski.NumHeteroatoms(mol) / mol.GetNumAtoms() for mol in tqdm(mols)
    ]
    # number of rings
    rings = [Lipinski.RingCount(mol) for mol in tqdm(mols)]
    ali_rings = [Lipinski.NumAliphaticRings(mol) for mol in tqdm(mols)]
    aro_rings = [Lipinski.NumAromaticRings(mol) for mol in tqdm(mols)]
    # hydrogen donors/acceptors
    h_donors = [Lipinski.NumHDonors(mol) for mol in mols]
    h_acceptors = [Lipinski.NumHAcceptors(mol) for mol in mols]
    # SA score
    SA = []
    for mol in tqdm(mols):
        try:
            SA.append(sascorer.calculateScore(mol))
        except (OverflowError, ZeroDivisionError):
            pass

    # NP-likeness
    fscore = npscorer.readNPModel()
    NP = [npscorer.scoreMol(mol, fscore) for mol in tqdm(mols)]

    # number of amide bonds
    n_amide = [CalcNumAmideBonds(mol) for mol in tqdm(mols)]
    # number of heterocycles
    n_heterocycles = [CalcNumHeterocycles(mol) for mol in tqdm(mols)]
    # number of bridgehead/spiro atoms
    n_bridgehead = [CalcNumBridgeheadAtoms(mol) for mol in tqdm(mols)]
    n_spiro = [CalcNumSpiroAtoms(mol) for mol in tqdm(mols)]

    # add all outcomes to data frame
    res.append(df.assign(outcome="Molecular weight", value=mws))
    res.append(df.assign(outcome="LogP", value=logp))
    res.append(df.assign(outcome="BertzTC", value=tcs))
    res.append(df.assign(outcome="TPSA", value=tpsa))
    res.append(df.assign(outcome="QED", value=qed))
    res.append(df.assign(outcome="% sp3 carbons", value=pct_sp3))
    res.append(df.assign(outcome="% rotatable bonds", value=pct_rot))
    res.append(df.assign(outcome="% stereocentres", value=pct_stereo))
    res.append(df.assign(outcome="% heteroatoms", value=pct_hetero))
    res.append(df.assign(outcome="# of rings", value=rings))
    res.append(df.assign(outcome="# of aliphatic rings", value=ali_rings))
    res.append(df.assign(outcome="# of aromatic rings", value=aro_rings))
    res.append(df.assign(outcome="# of hydrogen donors", value=h_donors))
    res.append(df.assign(outcome="# of hydrogen acceptors", value=h_acceptors))
    res.append(df.assign(outcome="# of amide bonds", value=n_amide))
    res.append(df.assign(outcome="# of heterocycles", value=n_heterocycles))
    res.append(df.assign(outcome="# of bridgehead atoms", value=n_bridgehead))
    res.append(df.assign(outcome="# of spiro atoms", value=n_spiro))
    res.append(df.assign(outcome="Synthetic accessibility score", value=SA))
    res.append(df.assign(outcome="Natural product-likeness score", value=NP))

    if "source" in list(df):
        # count heteroatoms separately for DeepMet vs. PubChem
        for src in df["source"].unique():
            src_idxs = np.where(df["source"] == src)[0]
            # TODO: was previously src_elements = np.take(elements, src_idxs)
            src_elements = [elements[i] for i in src_idxs]
            src_counts = np.unique(
                list(chain(*src_elements)), return_counts=True
            )
            for idx, element in enumerate(src_counts[0]):
                atom_count = src_counts[1][idx]
                res.append(
                    pd.DataFrame(
                        {
                            "source": src,
                            "outcome": "# atoms, " + element,
                            "value": [atom_count],
                        }
                    )
                )
    else:
        counts = np.unique(list(chain(*elements)), return_counts=True)
        for idx, element in enumerate(counts[0]):
            atom_count = counts[1][idx]
            res.append(
                pd.DataFrame(
                    {"outcome": "# atoms, " + element, "value": [atom_count]}
                )
            )

    # make output directories
    output_dir = os.path.dirname(output_file)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    res = pd.concat(res)
    # write output
    write_to_csv_file(output_file, res)
    res = res.reset_index(drop=True)
    return res


def main(args):
    calculate_outcome_distr(
        sample_file=args.sample_file,
        max_mols=args.max_mols,
        train_file=args.train_file,
        pubchem_file=args.pubchem_file,
        output_file=args.output_file,
    )
