"""
Calculate a set of outcomes for a list of SMILES, writing the complete
distribution and not just a summary statistic.
"""

import argparse
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
from clm.functions import clean_mol, pct_rotatable_bonds, pct_stereocenters, set_seed

# suppress Chem.MolFromSmiles error output
from rdkit import rdBase

rdBase.DisableLog("rdApp.error")


def add_args(parser):
    parser.add_argument("--input_file", type=str, help="Path to the input file.")
    parser.add_argument("--output_file", type=str, help="Path to the output file")
    return parser


def calculate_outcome_distr(input_file, output_file, seed=None):
    set_seed(seed)

    # create results container
    res = []

    # read SMILES and convert to molecules
    df = pd.read_csv(input_file)
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
            src_counts = np.unique(list(chain(*src_elements)), return_counts=True)
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
                pd.DataFrame({"outcome": "# atoms, " + element, "value": [atom_count]})
            )

    # make output directories
    output_dir = os.path.dirname(output_file)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    res = pd.concat(res)
    # write output
    res.to_csv(
        output_file,
        index=False,
        compression="gzip" if str(output_file).endswith(".gz") else None,
    )

    res = res.reset_index(drop=True)
    return res


def main(args):
    calculate_outcome_distr(input_file=args.input_file, output_file=args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
