import logging
import itertools
from tqdm import tqdm
import numpy as np
from rdkit.Chem import AllChem
from rdkit.DataStructs import FingerprintSimilarity
from selfies import encoder as selfies_encoder
from selfies.exceptions import EncoderError
from clm.functions import read_file, write_smiles, clean_mols
from clm.datasets import vocabulary_from_representation
from clm.util.SmilesEnumerator import SmilesEnumerator

import os
import pandas as pd

logger = logging.getLogger(__name__)


def add_args(parser):
    parser.add_argument(
        "--input-file", type=str, required=True, help="File path of smiles file"
    )
    parser.add_argument(
        "--train0-file",
        type=str,
        required=True,
        help="Output training smiles file path with no augmentation ({fold} in path is populated automatically)",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        required=True,
        help="Output training smiles file path ({fold} in path is populated automatically)",
    )
    parser.add_argument(
        "--vocab-file",
        type=str,
        required=True,
        help="Output training smiles vocabulary file path ({fold} in path is populated automatically)",
    )
    parser.add_argument(
        "--test0-file",
        type=str,
        required=True,
        help="Output test smiles file path with no augmentation ({fold} in path is populated automatically)",
    )
    parser.add_argument(
        "--enum-factor",
        type=int,
        default=0,
        help="For each input smile, number of randomized smiles to generate (0 for no augmentation)",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=10,
        help="Number of CV Folds to generate for train/test split (default %(default)s, 0 to not generate test data)",
    )
    parser.add_argument(
        "--which-fold",
        type=int,
        required=True,
        help="Which fold to generate? (0 to folds-1)",
    )
    parser.add_argument(
        "--representation",
        type=str,
        default="SMILES",
        help="Representation (one of SMILES/SELFIES)",
    )
    parser.add_argument(
        "--min-tc",
        type=float,
        default=0,
        help="Minimum fingerprint similarity (Tanimoto Coefficient) to seed molecule. 0 for no similarity requirement",
    )
    parser.add_argument(
        "--n-molecules",
        type=int,
        default=100,
        help="Number of molecules to generate for each seed molecule",
    )
    parser.add_argument(
        "--max-tries",
        type=int,
        default=200,
        help="Maximum tries to get n_molecules with min_tc",
    )
    parser.add_argument(
        "--max-input-smiles",
        type=int,
        default=None,
        help="Maximum smiles to read from input file (useful for testing)",
    )
    parser.add_argument(
        "--num-aug-neighbors",
        type=int,
        default=0,
        help="Number of augmented neighbors to add to training set",
    )
    parser.add_argument(
        "--min-tc-neighbors",
        type=float,
        default=0,
        help="Minimum Tanimoto Coefficient for augmented neighbors",
    )

    return parser


def get_similar_smiles(
    input_smiles, min_tc, n_molecules=100, max_tries=200, representation="SMILES"
):
    mols = clean_mols(input_smiles, selfies=representation == "SELFIES")
    input_smiles = [
        input_smiles[idx] for idx, mol in enumerate(mols) if mol is not None
    ]
    input_mols = [mol for mol in mols if mol is not None]
    logger.info(f"Calculating fingerprints for {len(input_mols)} valid molecules ...")
    input_fps = [
        AllChem.GetMorganFingerprintAsBitVect(input_mol, 3, nBits=1024)
        for input_mol in tqdm(input_mols)
    ]

    # shuffle SMILES and fingerprints
    inputs = list(zip(input_smiles, input_fps))
    np.random.shuffle(inputs)
    input_smiles, input_fps = zip(*inputs)

    # try to pick n molecules with minimum Tc to random seed molecule
    success = False
    for try_idx in range(max_tries):
        logger.info(
            f"picking {n_molecules} molecules with min_tc={min_tc} try #{try_idx} of {max_tries} ..."
        )
        inputs = list(zip(input_smiles, input_fps))
        np.random.shuffle(inputs)
        input_smiles, input_fps = zip(*inputs)

        # pick our seed molecule at random
        target_fp = input_fps[0]

        tcs = [FingerprintSimilarity(input_fp, target_fp) for input_fp in input_fps]
        # subset SMILES based on fingerprint similarity
        subset_smiles = [
            input_smiles for input_smiles, tc in zip(input_smiles, tcs) if tc >= min_tc
        ]

        # break if we have enough molecules
        if len(subset_smiles) >= n_molecules:
            subset_smiles = subset_smiles[: int(n_molecules)]
            success = True
            break

    if not success:
        raise RuntimeError(
            f"Unable to pick {n_molecules} molecules with min_tc={min_tc}"
        )

    return subset_smiles


# def create_training_sets(
#     input_file=None,
#     train0_file=None,
#     train_file=None,
#     test0_file=None,
#     vocab_file=None,
#     folds=10,
#     which_fold=0,
#     enum_factor=0,
#     representation="SMILES",
#     min_tc=0,
#     n_molecules=100,
#     max_tries=200,
#     max_input_smiles=None,
# ):
#     logger.info("reading input SMILES ...")
#     data = read_file(
#         smiles_file=input_file, smile_only=True, max_lines=max_input_smiles
#     )
#     smiles = data["smiles"]

#     if min_tc > 0:
#         logger.info(f"picking {n_molecules} molecules with min_tc={min_tc} ...")
#         smiles = get_similar_smiles(
#             smiles,
#             min_tc=min_tc,
#             n_molecules=n_molecules,
#             max_tries=max_tries,
#             representation=representation,
#         )

#     generate_test_data = folds > 0
#     if generate_test_data:
#         np.random.shuffle(smiles)
#         folds = np.array_split(smiles, folds)
#     else:
#         folds = [smiles]

#     if enum_factor > 0:
#         enum_folds = [np.array([]) for i in range(len(folds))]
#         sme = SmilesEnumerator(canonical=False, enum=True)
#         for idx, fold in enumerate(folds):
#             enum = []
#             max_tries = 200  # randomized SMILES to generate for each input structure
#             for sm_idx, sm in enumerate(tqdm(fold)):
#                 tries = []
#                 for try_idx in range(max_tries):
#                     try:
#                         this_try = sme.randomize_smiles(sm)
#                         tries.append(this_try)
#                         tries = [rnd for rnd in np.unique(tries)]
#                         if len(tries) > enum_factor:
#                             tries = tries[:enum_factor]
#                             break
#                     except AttributeError:
#                         continue
#                 enum.extend(tries)
#             enum_folds[idx] = enum
#     else:
#         enum_folds = folds

#     if generate_test_data:
#         test0 = folds[which_fold]
#         test = enum_folds[which_fold]
#         train0 = folds[:which_fold] + folds[which_fold + 1 :]
#         train0 = list(itertools.chain.from_iterable(train0))
#         train = enum_folds[:which_fold] + enum_folds[which_fold + 1 :]
#         train = list(itertools.chain.from_iterable(train))
#     else:
#         train0 = folds[0]
#         train = enum_folds[0]
#         test0 = None
#         test = None

#     if representation == "SELFIES":
#         logger.info("converting SMILES strings to SELFIES ...")

#         train0_out = []
#         for sm in train0:
#             try:
#                 sf = selfies_encoder(sm)
#                 train0_out.append(sf)
#             except EncoderError:
#                 pass
#         train0 = train0_out

#         train_out = []
#         for sm in train:
#             try:
#                 sf = selfies_encoder(sm)
#                 train_out.append(sf)
#             except EncoderError:
#                 pass
#         train = train_out

#         if test0 is not None:
#             test0_out = []
#             for sm in test0:
#                 try:
#                     sf = selfies_encoder(sm)
#                     test0_out.append(sf)
#                 except EncoderError:
#                     pass
#             test0 = test0_out

#         if test is not None:
#             test_out = []
#             for sm in test:
#                 try:
#                     sf = selfies_encoder(sm)
#                     test_out.append(sf)
#                 except EncoderError:
#                     pass
#             test = test_out

#     write_smiles(
#         train0,
#         str(train0_file).format(fold=which_fold),
#         add_inchikeys=True,
#         extra_data=data,
#     )
#     write_smiles(
#         train,
#         str(train_file).format(fold=which_fold),
#         add_inchikeys=True,
#         extra_data=data,
#     )
#     vocabulary = vocabulary_from_representation(representation, train)
#     logger.info("vocabulary of {} characters".format(len(vocabulary)))
#     vocabulary.write(output_file=str(vocab_file).format(fold=which_fold))
#     if test0 is not None:
#         write_smiles(
#             test0,
#             str(test0_file).format(fold=which_fold),
#             add_inchikeys=True,
#             extra_data=data,
#         )

def create_training_sets(
    input_file=None,
    train0_file=None,
    train_file=None,
    test0_file=None,
    vocab_file=None,
    folds=10,
    which_fold=0,
    enum_factor=0,
    representation="SMILES",
    min_tc=0,
    n_molecules=100,
    max_tries=200,
    max_input_smiles=None,
    num_aug_neighbors=0,
    min_tc_neighbors=0
):
    logger.info("reading input SMILES ...")
    data = read_file(
        smiles_file=input_file, smile_only=True, max_lines=max_input_smiles
    )
    smiles = data["smiles"]

    if min_tc > 0:
        logger.info(f"picking {n_molecules} molecules with min_tc={min_tc} ...")
        smiles = get_similar_smiles(
            smiles,
            min_tc=min_tc,
            n_molecules=n_molecules,
            max_tries=max_tries,
            representation=representation,
        )

    generate_test_data = folds > 0
    if generate_test_data:
        np.random.shuffle(smiles)
        folds = np.array_split(smiles, folds)
    else:
        folds = [smiles]

    import glob
    from rdkit import rdBase, Chem
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.DataStructs import TanimotoSimilarity

    # set working directory
    import sys
    sys.path.append("/Genomics/skinniderlab/spavelites/git/PED-generation/python")
    import functions

    in_dir = "/Genomics/skinniderlab/spavelites/git/PED-generation/outputs/pubchem/all_splits_pr"
    csv_files = glob.glob(os.path.join(in_dir, "*.csv"))
    big_data = []
    top_n = num_aug_neighbors
    min_tc_2 = min_tc_neighbors/100 #min_tc taking as another variable to avoid confusion

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    for f in tqdm(csv_files, desc="Processing CSV files"): # REMOVE [:3] POST TESTING!!!!
        df = pd.read_csv(f).head(3)
        df["neighbor_smiles"] = df["neighbor_smiles"].apply(lambda x: functions.preprocess_mol(x)[1] if pd.notnull(x) else None)
        df = df[df["neighbor_smiles"] != df["target_smiles"]]
        if df.empty:
            continue
        tfp = gen.GetFingerprint(Chem.MolFromSmiles(df["target_smiles"].iloc[0]))
        df["tanimoto"] = [
            TanimotoSimilarity(tfp, gen.GetFingerprint(Chem.MolFromSmiles(smi)))
            for smi in df["neighbor_smiles"]
        ]
        # df = df[df["tanimoto"] < 1.0].dropna().head(top_n)
        df = df[(df["tanimoto"] >= min_tc_2) & (df["tanimoto"] < 1.0)].dropna().head(top_n)
        lines = df["neighbor_smiles"].tolist()
        if not df.empty:
            row = [df["target_smiles"].iloc[0]] + lines
            big_data.append(row)

    col_names = ["target_smiles"] + [f"neighbor_smiles_{i+1}" for i in range(top_n)]
    big_df = pd.DataFrame(big_data, columns=col_names)
    print(big_df.head())

    if generate_test_data:
        test0 = folds[which_fold]
        test = folds[which_fold]
        train0 = folds[:which_fold] + folds[which_fold + 1 :]
        train0 = list(itertools.chain.from_iterable(train0))
        # add neighbor_smiles from big_df for matching target_smiles in train0
        train0_extended = []
        for sm in train0:
            train0_extended.append(sm)
            matched_rows = big_df[big_df["target_smiles"] == sm]
            if not matched_rows.empty:
                for _, row in matched_rows.iterrows():
                    neighbor_cols = [c for c in row.index if c.startswith("neighbor_smiles_")]
                    neighbors = [row[c] for c in neighbor_cols if pd.notnull(row[c])]
                    train0_extended.extend(neighbors)
        train0 = train0_extended

        train = train0

    else:
        train0 = folds[0]
        train = folds[0]
        test0 = None
        test = None

    # now incorporate enumeration if enum_factor > 0
    if enum_factor > 0:
        sme = SmilesEnumerator(canonical=False, enum=True)

        augmented_train = []
        for sm_idx, sm in enumerate(tqdm(train)):
            tries = []
            for _ in range(200):
                try:
                    this_try = sme.randomize_smiles(sm)
                    tries.append(this_try)
                    tries = [rnd for rnd in np.unique(tries)]
                    if len(tries) >= enum_factor:
                        break
                except AttributeError:
                    continue
            augmented_train.extend(tries)
        train = augmented_train



    else:
        train0 = folds[0]
        train = enum_folds[0]
        test0 = None
        test = None

    write_smiles(
        train0,
        str(train0_file).format(fold=which_fold),
        add_inchikeys=True,
        extra_data=data,
    )
    write_smiles(
        train,
        str(train_file).format(fold=which_fold),
        add_inchikeys=True,
        extra_data=data,
    )
    vocabulary = vocabulary_from_representation(representation, train)
    logger.info("vocabulary of {} characters".format(len(vocabulary)))
    vocabulary.write(output_file=str(vocab_file).format(fold=which_fold))
    if test0 is not None:
        write_smiles(
            test0,
            str(test0_file).format(fold=which_fold),
            add_inchikeys=True,
            extra_data=data,
        )


def main(args):
    create_training_sets(
        input_file=args.input_file,
        train0_file=args.train0_file,
        train_file=args.train_file,
        test0_file=args.test0_file,
        vocab_file=args.vocab_file,
        folds=args.folds,
        which_fold=args.which_fold,
        enum_factor=args.enum_factor,
        representation=args.representation,
        min_tc=args.min_tc,
        n_molecules=args.n_molecules,
        max_tries=args.max_tries,
        max_input_smiles=args.max_input_smiles,
        num_aug_neighbors=args.num_aug_neighbors,
        min_tc_neighbors=args.min_tc_neighbors
    )
