"""
Apply the Renz et al. 'AddCarbon' model to the training set.
"""

import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from tqdm import tqdm
import pandas as pd

# import functions
from clm.functions import (
    clean_mol,
    write_smiles,
    read_file,
    read_csv_file,
    write_to_csv_file,
)


def add_args(parser):
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    return parser


def add_carbon(input_file, output_file):
    # make output directories
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    # remove output file if it exists
    if os.path.exists(output_file):
        os.remove(output_file)

    data = []

    # read the input SMILES
    dataframe = read_file(input_file, smile_only=False)
    smiles = dataframe["smiles"].tolist()
    if "inchikey" in dataframe.columns:
        train_inchi = dataframe["inchikey"].tolist()
    else:
        train_mols = [clean_mol(smile, raise_error=False) for smile in smiles]
        train_inchi = set(
            [Chem.inchi.MolToInchiKey(mol) for mol in train_mols if mol]
        )

    # loop over the input SMILES
    # output_smiles = list()
    for sm_idx, input_smiles in enumerate(tqdm(smiles)):
        print(
            "working on SMILES {} of {}: '{}' ...".format(
                sm_idx, len(smiles), input_smiles
            )
        )
        """
        code adapted from:
        https://github.com/ml-jku/mgenerators-failure-modes/blob/master/addcarbon.py
        """
        # try all positions in the molecule in random order
        for i in np.random.permutation(len(input_smiles)):
            # insert C at a random spot and check if valid
            mut = input_smiles[:i] + "C" + input_smiles[i:]
            try:
                mut_mol = clean_mol(mut)
            except Exception:
                continue
            # catch #2
            if mut_mol is None:
                continue

            # if it is valid, compute canonical smiles
            mut_can = Chem.MolToSmiles(mut_mol, isomericSmiles=False)
            mut_inchi = Chem.inchi.MolToInchiKey(mut_mol)
            # can't be in the training set
            if mut_inchi in train_inchi:
                continue

            # calculate exact mass
            exact_mass = Descriptors.ExactMolWt(mut_mol)
            # round to 6 decimal places
            mass = round(exact_mass, 6)

            # calculate molecular formula
            formula = rdMolDescriptors.CalcMolFormula(mut_mol)

            # append to file
            data.append([input_smiles, mut_can, str(mass), formula, mut_inchi])

        # see if we can break
        # if len(output_smiles) > args.max_smiles:
        #     break

    df = pd.DataFrame(
        columns=[
            "input_smiles",
            "mutated_smiles",
            "mass",
            "formula",
            "inchikey",
        ],
        dtype=str,
        data=data,
    )
    write_to_csv_file(output_file, df)

    # write unique SMILES
    uniq_smiles = read_csv_file(output_file).mutated_smiles.unique()
    filename = str(output_file).split(os.extsep)[0]
    uniq_file = filename + "-unique.smi"
    write_smiles(uniq_smiles, uniq_file)


def main(args):
    add_carbon(input_file=args.input_file, output_file=args.output_file)
