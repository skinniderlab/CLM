import contextlib

import deepsmiles
import numpy as np
import os
import os.path
import pandas as pd
import warnings
from selfies import decoder
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, Lipinski, rdmolops, Descriptors, rdMolDescriptors
from rdkit.DataStructs import FingerprintSimilarity
import torch
from scipy import histogram
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon
import hashlib
import gzip
import logging
import csv


logger = logging.getLogger(__name__)
converter = deepsmiles.Converter(rings=True, branches=True)


def clean_mol(
    smiles, *, stereochem=False, selfies=False, deepsmiles=False, raise_error=True
):
    """
    Construct a molecule from a SMILES string, removing stereochemistry and
    explicit hydrogens, and setting aromaticity.
    """
    if selfies:
        selfies = smiles.replace("<PAD>", "[nop]")
        smiles = decoder(selfies)
    elif deepsmiles:
        deepsmiles = smiles
        try:
            smiles = converter.decode(deepsmiles)
        except ValueError:
            raise ValueError(f"invalid DeepSMILES: {deepsmiles}")
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        if raise_error:
            raise ValueError("invalid SMILES: " + str(smiles))
        else:
            return None
    if not stereochem:
        Chem.RemoveStereochemistry(mol)
    Chem.SanitizeMol(mol)
    mol = Chem.RemoveHs(mol)
    return mol


def clean_mols(
    all_smiles,
    *,
    stereochem=False,
    selfies=False,
    deepsmiles=False,
    disable_progress=False,
    return_dict=False,
):
    """
    Construct a list of molecules from a list of SMILES strings, replacing
    invalid molecules with None in the list.
    """
    mols = {}
    for smile in tqdm(all_smiles, disable=disable_progress):
        try:
            mol = clean_mol(
                smile, stereochem=stereochem, selfies=selfies, deepsmiles=deepsmiles
            )
            mols[smile] = mol
        except ValueError:
            mols[smile] = None

    if return_dict:
        return mols
    else:
        return list(mols.values())


def remove_salts_solvents(mol, hac=3):
    """
    Remove solvents and ions have max 'hac' heavy atoms.
    This function was obtained from the mol2vec package,
    available at:
        https://github.com/samoturk/mol2vec/blob/master/mol2vec/features.py
    """
    # split molecule into fragments
    fragments = list(rdmolops.GetMolFrags(mol, asMols=True))
    # keep heaviest only
    # fragments.sort(reverse=True, key=lambda m: m.GetNumAtoms())
    # remove fragments with < 'hac' heavy atoms
    fragments = [fragment for fragment in fragments if fragment.GetNumAtoms() > hac]
    #
    if len(fragments) > 1:
        warnings.warn(
            "molecule contains >1 fragment with >" + str(hac) + " heavy atoms"
        )
        return None
    elif len(fragments) == 0:
        warnings.warn(
            "molecule contains no fragments with >" + str(hac) + " heavy atoms"
        )
        return None
    else:
        return fragments[0]


def compute_fingerprints(mols, algorithm="rdkit", include_none=False):
    """
    Get ECFP6/ RDKIT fingerprints for a list of molecules. Optionally,
    handle `None` values by including a `None` value in that
    position.
    """
    fps = []
    for mol in mols:
        if mol is None and include_none:
            fps.append(None)
        else:
            fp = compute_fingerprint(mol, algorithm=algorithm)
            fps.append(fp)
    return fps


def compute_fingerprint(mol, algorithm="rdkit"):
    if algorithm == "rdkit":
        return Chem.RDKFingerprint(mol)
    elif algorithm == "ecfp6":
        return AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024)
    else:
        raise ValueError("Unsupported fingerprint algorithm specified")


def get_column_idx(input_file, column_name):
    with open(input_file, "r") as f:
        first_line = f.readline().strip()
        idx = first_line.split(",").index(column_name)

    return idx


def read_file(
    smiles_file, max_lines=None, smile_only=False, stream=False, randomize=False
):
    """
    Read a line-delimited file of SMILES strings, optionally limiting the number
    of lines read and/or returning only the SMILES strings themselves.
    Randomization, if enabled, will shuffle the lines in the file before reading.

    Args:
        smiles_file: Input file containing SMILES strings, or comma-separated file with "smiles" in the header.
        max_lines: Maximum number of lines to return.
        smile_only: Whether to return only the SMILES strings.
        stream: Whether to return a generator or a list.
        randomize: Whether to shuffle the lines in the file before reading.

    Returns:
        An iterator or list of strings.
    """

    def _read_file(
        input_file,
        max_lines=None,
        smile_only=False,
    ):
        if str(input_file).endswith(".gz"):
            open_fn = gzip.open
            mode = "rt"
        else:
            open_fn = open
            mode = "r"

        count = 0
        with open_fn(input_file, mode) as f:
            # Detect if we're dealing with a csv file with "smiles" in the header
            first_line = f.readline()
            first_line_tokens = next(csv.reader([first_line]))
            is_csv = "smiles" in first_line_tokens

            if is_csv:
                smile_idx = first_line_tokens.index("smiles")
            else:
                smile_idx = None
                f.seek(0)  # go to beginning of file

            for line in f:
                tokens = next(csv.reader([line]))
                if is_csv and smile_only:
                    yield tokens[smile_idx]
                else:
                    yield line.strip()
                count += 1
                if max_lines is not None and count == max_lines:
                    return

    if randomize:
        # if randomizing, we have to consume the generator and shuffle it
        gen = _read_file(smiles_file, max_lines=None, smile_only=smile_only)
        data = np.array(list(gen))
        np.random.shuffle(data)
        data = data[:max_lines]
        return iter(data) if stream else data
    else:
        gen = _read_file(smiles_file, max_lines, smile_only)
        return gen if stream else np.array(list(gen))


def write_smiles(smiles, smiles_file, mode="w", add_inchikeys=False):
    """
    Write a list of SMILES to a line-delimited file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(smiles_file)), exist_ok=True)
    with open(smiles_file, mode) as f:
        if add_inchikeys:
            f.write("smiles,inchikey\n")
        for sm in smiles:
            f.write(sm)
            if add_inchikeys:
                if mol := clean_mol(sm, raise_error=False):
                    inchikey = Chem.inchi.MolToInchiKey(mol)
                else:
                    inchikey = ""
                f.write(f",{inchikey}")
            f.write("\n")


"""
rdkit contributed code to neutralize charged molecules;
obtained from:
    https://www.rdkit.org/docs/Cookbook.html
    http://www.mail-archive.com/rdkit-discuss@lists.sourceforge.net/msg02669.html
"""


def _InitialiseNeutralisationReactions():
    patts = (
        # Imidazoles
        ("[n+;H]", "n"),
        # Amines
        ("[N+;!H0]", "N"),
        # Carboxylic acids and alcohols
        ("[$([O-]);!$([O-][#7])]", "O"),
        # Thiols
        ("[S-;X1]", "S"),
        # Sulfonamides
        ("[$([N-;X2]S(=O)=O)]", "N"),
        # Enamines
        ("[$([N-;X2][C,N]=C)]", "N"),
        # Tetrazoles
        ("[n-]", "[nH]"),
        # Sulfoxides
        ("[$([S-]=O)]", "S"),
        # Amides
        ("[$([N-]C=O)]", "N"),
    )
    return [(Chem.MolFromSmarts(x), Chem.MolFromSmiles(y, False)) for x, y in patts]


_reactions = None


def NeutraliseCharges(mol, reactions=None):
    global _reactions
    if reactions is None:
        if _reactions is None:
            _reactions = _InitialiseNeutralisationReactions()
        reactions = _reactions
    for i, (reactant, product) in enumerate(reactions):
        while mol.HasSubstructMatch(reactant):
            rms = AllChem.ReplaceSubstructs(mol, reactant, product)
            mol = rms[0]
    return mol


def set_seed(seed):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)


def seed_type(value):
    # A "type" useful for argparse arguments for random seeds
    # that can come in as "None" (e.g. from a snakemake workflow)
    return None if value == "None" else int(value)


def continuous_JSD(generated_dist, original_dist, tol=1e-10):
    # Remove none values from either distributions
    original_dist = original_dist[~original_dist.isna()]
    generated_dist = generated_dist[~generated_dist.isna()]

    if len(generated_dist) < 2:  # not enough points?
        return np.nan
    try:
        gen_kde = gaussian_kde(generated_dist)
        org_kde = gaussian_kde(original_dist)
    except np.linalg.LinAlgError:
        return np.nan

    vec = np.hstack([generated_dist, original_dist])
    x_eval = np.linspace(vec.min(), vec.max(), num=1000)
    P = gen_kde(x_eval) + tol
    Q = org_kde(x_eval) + tol
    return jensenshannon(P, Q)


def discrete_JSD(generated_dist, original_dist, tol=1e-10):
    min_v = min(min(generated_dist), min(original_dist))
    max_v = max(max(generated_dist), max(original_dist))
    gen, bins = histogram(generated_dist, bins=range(min_v, max_v + 1, 1), density=True)
    org, bins = histogram(original_dist, bins=range(min_v, max_v + 1, 1), density=True)
    gen += tol
    org += tol
    return jensenshannon(gen, org)


def internal_diversity(fps, sample_size=1e4, summarise=True):
    """
    Calculate the internal diversity, defined as the mean intra-set Tanimoto
    coefficient, between a set of fingerprints. For large sets, calculating the
    entire matrix is prohibitive, so a random set of molecules are sampled.
    """
    tcs = []
    counter = 0
    while counter < sample_size:
        idx1 = np.random.randint(0, len(fps))
        idx2 = np.random.randint(0, len(fps))
        fp1 = fps[idx1]
        fp2 = fps[idx2]
        tcs.append(FingerprintSimilarity(fp1, fp2))
        counter += 1
    if summarise:
        return np.mean(tcs)
    else:
        return tcs


def external_diversity(fps1, fps2, sample_size=1e4, summarise=True):
    """
    Calculate the external diversity, defined as the mean inter-set Tanimoto
    coefficient, between two sets of fingerprints. For large sets, calculating
    the entire matrix is prohibitive, so a random set of molecules are sampled.
    """
    tcs = []
    counter = 0
    while counter < sample_size:
        idx1 = np.random.randint(0, len(fps1))
        idx2 = np.random.randint(0, len(fps2))
        fp1 = fps1[idx1]
        fp2 = fps2[idx2]
        tcs.append(FingerprintSimilarity(fp1, fp2))
        counter += 1
    if summarise:
        if len(tcs) == 0:
            return np.nan
        else:
            return np.mean(tcs)
    else:
        return tcs


def internal_nn(fps, sample_size=1e3, summarise=True):
    """
    Calculate the nearest-neighbor Tanimoto coefficient within a set of
    fingerprints.
    """
    counter = 0
    nns = []
    while counter < sample_size:
        idx1 = np.random.randint(0, len(fps))
        fp1 = fps[idx1]
        tcs = []
        for idx2 in range(len(fps)):
            if idx1 != idx2:
                fp2 = fps[idx2]
                tcs.append(FingerprintSimilarity(fp1, fp2))

        if len(tcs) == 0:  # not enough fingerprints?
            return np.nan

        nn = np.max(tcs)
        nns.append(nn)
        counter += 1
    if summarise:
        if len(nns) == 0:
            return np.nan
        else:
            return np.mean(nns)
    else:
        return nns


def external_nn(fps1, fps2, sample_size=1e3, summarise=True):
    """i
    Calculate the nearest-neighbor Tanimoto coefficient, searching one set of
    fingerprints against a second set.
    """
    counter = 0
    nns = []
    while counter < sample_size:
        idx1 = np.random.randint(0, len(fps1))
        fp1 = fps1[idx1]
        tcs = []
        for idx2 in range(len(fps2)):
            fp2 = fps2[idx2]
            tcs.append(FingerprintSimilarity(fp1, fp2))

        if len(tcs) == 0:  # not enough fingerprints?
            return np.nan

        nn = np.max(tcs)
        nns.append(nn)
        counter += 1
    if summarise:
        return np.mean(nns)
    else:
        return nns


def pct_rotatable_bonds(mol):
    n_bonds = mol.GetNumBonds()
    if n_bonds > 0:
        rot_bonds = Lipinski.NumRotatableBonds(mol) / n_bonds
    else:
        rot_bonds = 0
    return rot_bonds


def pct_stereocenters(mol):
    n_atoms = mol.GetNumAtoms()
    if n_atoms > 0:
        Chem.AssignStereochemistry(mol)
        pct_stereo = AllChem.CalcNumAtomStereoCenters(mol) / n_atoms
    else:
        pct_stereo = 0
    return pct_stereo


def generate_df(smiles_file, chunk_size):
    smiles_df = read_csv_file(smiles_file)
    smiles = smiles_df["smiles"].to_list()
    df = pd.DataFrame(columns=["smiles", "mass", "formula"])

    for i in tqdm(range(0, len(smiles), chunk_size)):
        mols = clean_mols(
            smiles[i : i + chunk_size],
            selfies=False,
            disable_progress=True,
            return_dict=True,
        )

        chunk_data = [
            {
                "smiles": smile,
                "mass": round(Descriptors.ExactMolWt(mol), 4),
                "formula": rdMolDescriptors.CalcMolFormula(mol),
            }
            for smile, mol in mols.items()
            if mol
        ]

        if chunk_data:
            df = pd.concat([df, pd.DataFrame(chunk_data)], ignore_index=True)

    # Add any additional columns from the input file
    df = df.merge(smiles_df, how="left", on="smiles")
    return df


def get_mass_range(mass, err_ppm):
    min_mass = (-err_ppm / 1e6 * mass) + mass
    max_mass = (err_ppm / 1e6 * mass) + mass

    return min_mass, max_mass


def write_to_csv_file(filepath, info, mode="w", header=True, columns=None):
    assert mode in ("w", "a+"), "Invalid mode specified"

    # os.path.dirname(filepath) returns '' if filepath is just a filename.
    if (basedir := os.path.dirname(filepath)) != "":
        os.makedirs(basedir, exist_ok=True)

    compression = "gzip" if str(filepath).endswith(".gz") else None

    # See if the provided information is a dataframe
    if isinstance(info, pd.DataFrame):
        info.to_csv(
            filepath,
            mode=mode,
            header=header if mode == "w" else False,
            columns=columns,
            index=False,
            compression=compression,
        )
    else:
        raise RuntimeError("only DataFrame input is supported")


def read_csv_file(filename, **kwargs):
    compression = "gzip" if str(filename).endswith(".gz") else None
    return pd.read_csv(filename, compression=compression, **kwargs)


def assert_checksum_equals(generated_file, oracle):
    assert (
        hashlib.md5(
            "".join(open(generated_file, "r").readlines()).encode("utf8")
        ).hexdigest()
        == hashlib.md5(
            "".join(open(oracle, "r").readlines()).encode("utf8")
        ).hexdigest()
    )


def split_frequency_ranges(data, max_molecules=None, all=False):
    # Reset index of the dataframe before doing anything
    data = data.reset_index(drop=True)

    # TODO: make this process dynamic later
    frequency_ranges = [(1, 1), (2, 2), (3, 10), (11, 30), (31, 100), (101, None)]

    data["bin"] = ""
    for (f_min, f_max) in frequency_ranges:
        if f_max is not None:
            selected_rows = data[data["size"].between(f_min, f_max)]
        else:
            selected_rows = data[data["size"] >= f_min]

        bin_name = f"{f_min}-{f_max}" if f_max is not None else f"{f_min}-"

        if max_molecules is not None and len(selected_rows) > max_molecules:
            selected_rows = selected_rows.sample(n=max_molecules)
        elif max_molecules is not None and len(selected_rows) < max_molecules:
            logger.warning(
                f"Not enough molecules for frequency bin '{bin_name}'. Using {len(selected_rows)} molecules."
            )

        data.loc[selected_rows.index, "bin"] = bin_name

    if all:
        # Sample non-binned df by weight
        if max_molecules is not None:
            if len(data) >= max_molecules:
                selected_rows = data.sample(n=max_molecules, weights="size")
            else:
                logger.warning(
                    f"Not enough molecules for {max_molecules}, using {len(data)} instead."
                )
                selected_rows = data.copy()
        else:
            selected_rows = data.copy()

        selected_rows["bin"] = "all"
        data = pd.concat([data, selected_rows])

    # Save only the rows where we've assigned a bin
    data = data[data["bin"] != ""].reset_index(drop=True)

    return data


@contextlib.contextmanager
def local_seed(seed):
    current_state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(current_state)
