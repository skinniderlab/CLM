import deepsmiles
import numpy as np
import os
import os.path
import random
import warnings
from selfies import decoder
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, Lipinski, rdmolops
from rdkit.DataStructs import FingerprintSimilarity
import torch
from scipy import histogram
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon

converter = deepsmiles.Converter(rings=True, branches=True)


def clean_mol(smiles, *, stereochem=False, selfies=False, deepsmiles=False):
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
        raise ValueError("invalid SMILES: " + str(smiles))
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


def get_ecfp6_fingerprints(mols, include_none=False):
    """
    Get ECFP6 fingerprints for a list of molecules. Optionally,
    handle `None` values by returning a `None` value in that
    position.
    """
    fps = []
    for mol in mols:
        if mol is None and include_none:
            fps.append(None)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024)
            fps.append(fp)
    return fps


def get_rdkit_fingerprints(mols, include_none=False):
    """
    Get RDKIT fingerprints for a list of molecules. Optionally,
    handle `None` values by returning a `None` value in that
    position.
    """
    fps = []
    for mol in mols:
        if mol is None and include_none:
            fps.append(None)
        else:
            fp = Chem.RDKFingerprint(mol)
            fps.append(fp)
    return fps


def read_smiles(smiles_file, max_lines=None):
    """
    Read a list of SMILES from a line-delimited file.
    """
    with open(smiles_file, "r") as f:
        lines = np.array(f.readlines())

    lines = np.char.strip(lines)

    if max_lines is not None and max_lines > 0:
        lines = lines[:max_lines]

    return lines


def write_smiles(smiles, smiles_file, mode="w"):
    """
    Write a list of SMILES to a line-delimited file.
    """
    os.makedirs(os.path.dirname(smiles_file), exist_ok=True)
    with open(smiles_file, mode) as f:
        for sm in smiles:
            _ = f.write(sm + "\n")


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
    try:
        gen_kde = gaussian_kde(generated_dist)
    except np.linalg.LinAlgError:
        generated_dist += np.random.normal(0, 1e-5, len(generated_dist))
        gen_kde = gaussian_kde(generated_dist)

    org_kde = gaussian_kde(original_dist)
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
        idx1 = random.randint(0, len(fps) - 1)
        idx2 = random.randint(0, len(fps) - 1)
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
    #
    tcs = []
    counter = 0
    while counter < sample_size:
        idx1 = random.randint(0, len(fps1) - 1)
        idx2 = random.randint(0, len(fps2) - 1)
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
        idx1 = random.randint(0, len(fps) - 1)
        fp1 = fps[idx1]
        tcs = []
        for idx2 in range(len(fps)):
            if idx1 != idx2:
                fp2 = fps[idx2]
                tcs.append(FingerprintSimilarity(fp1, fp2))
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
        idx1 = random.randint(0, len(fps1) - 1)
        fp1 = fps1[idx1]
        tcs = []
        for idx2 in range(len(fps2)):
            fp2 = fps2[idx2]
            tcs.append(FingerprintSimilarity(fp1, fp2))
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
