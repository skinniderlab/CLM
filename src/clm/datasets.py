"""
Datasets used by PyTorch for language modelling of chemical structures.
"""

import numpy as np
import re
import selfies as sf
import torch
from torch.nn.utils.rnn import pad_sequence
from itertools import chain
from torch.utils.data import Dataset
from rdkit.Chem import Descriptors, rdMolDescriptors, QED
from clm.functions import read_file, clean_mol


class SmilesCollate:
    """
    Collate a list of SMILES tensors, with variable lengths, into a tensor.

    Code adapted from: https://www.codefull.org/2018/11/use-pytorchs-\
    dataloader-with-variable-length-sequences-for-lstm-gru/

    Args:
        batch (list): a list of numeric tensors, each derived from a single
        SMILES string, where the value at each position in the tensor
        is the index of the SMILES token in the vocabulary dictionary

    Return:
        a tensor of dimension (batch_size, seq_len) containing encoded and
          padded sequences
    """

    def __init__(self, vocabulary, descriptors=None):
        self.padding_token = vocabulary.dictionary["<PAD>"]
        self.descriptors = descriptors

    def __call__(self, smiles_encoded):
        _, encoded, _ = zip(*smiles_encoded)
        padded = pad_sequence(encoded, padding_value=self.padding_token)
        lengths = [len(seq) for seq in encoded]
        return padded, lengths


class SmilesDescriptorsCollate(SmilesCollate):
    # functions that take in a molecule and return a scalar
    descriptor_fns = [
        Descriptors.ExactMolWt,
        Descriptors.MolLogP,
        Descriptors.TPSA,
        rdMolDescriptors.CalcNumHBA,
        rdMolDescriptors.CalcNumHBD,
        QED.qed,
    ]
    descriptor_names = [
        "Molecular Weight",
        "Wildman– Crippen partition coefficient (LogP)",
        "Topological polar surface area (TPSA)",
        "Number of Hydrogen bonds acceptors (HBA)",
        "Number of Hydrogen bonds donors (HBD)",
        "Drug-likeness score (QED)",
    ]

    def __init__(self, vocabulary, descriptors=None):
        super().__init__(vocabulary)
        self.descriptors = descriptors

    @classmethod
    def get_descriptors(cls, smiles):  # Generate descriptors dynamically
        values = []
        for sm in smiles:
            if mol := clean_mol(sm, raise_error=False):
                value = [fn(mol) for fn in cls.descriptor_fns]
            else:
                value = [0] * len(cls.descriptor_fns)
            values.append(value)
        return values

    @property
    def n_descriptors(self):
        if self.descriptors is not None:
            return self.descriptors.shape[1]
        return len(self.descriptor_fns)

    @property
    def dynamic_descriptors(self):
        return (
            self.descriptor_names
        )  # Column names for the dynamic descriptors [Avoid changing the order]

    def __call__(self, smiles_encoded):
        padded, lengths = super().__call__(smiles_encoded)
        smiles, _, descriptors = zip(*smiles_encoded)
        if descriptors is not None and not all(
            desc is None for desc in descriptors
        ):  # Checking if there was a file with descriptors being used or not
            descriptors = torch.tensor(np.array(descriptors), dtype=torch.float32)
        else:
            descriptors = torch.tensor(self.get_descriptors(smiles))
        return padded, lengths, descriptors


class SmilesDataset(Dataset):
    """
    A dataset of chemical structures, provided in SMILES format.
    """

    def __init__(
        self,
        smiles,
        max_len=None,
        vocab_file=None,
        training_split=0.9,
        collate_class=SmilesCollate,
        descriptors=None,
    ):
        """
        Can be initiated from either a list of SMILES, or a line-delimited
        file.

        Args:
            smiles (list): the complete set of SMILES that constitute the
              training dataset
            smiles_file (string): line-delimited file containing the complete
              set of SMILES that constitute the training dataset
            vocab_file (string): line-delimited file containing all tokens to
              be used in the vocabulary
            training_split (numeric): proportion of the dataset to withhold for
              validation loss calculation
        """

        # shuffle the SMILES
        self.smiles = smiles
        np.random.shuffle(self.smiles)

        # create vocabulary or else read from file
        if vocab_file:
            self.vocabulary = Vocabulary(vocab_file=vocab_file)
        else:
            self.vocabulary = Vocabulary(smiles=self.smiles)

        # remove SMILES greater than max_len
        self.max_len = max_len
        if self.max_len is not None:
            self.smiles = [
                sm
                for sm in self.smiles
                if len(self.vocabulary.tokenize(sm)) <= self.max_len
            ]

        # split out a validation set
        n_smiles = len(self.smiles)
        border = int(n_smiles * training_split)
        self.training_set = self.smiles[:border]
        self.validation_set = self.smiles[border:]
        self.descriptors = False
        # Check for descriptors if present create a training and test split
        if descriptors is not None:
            if isinstance(descriptors, np.ndarray) and not np.any(
                np.isnan(descriptors)
            ):
                self.descriptors = True
                self.descriptors_training_set = descriptors[:border]
                self.descriptors_validation_set = descriptors[border:]
            else:
                self.descriptors = False

        self.collate = collate_class(self.vocabulary, descriptors=descriptors)

    def __len__(self):
        return len(self.training_set)

    def __getitem__(self, idx):
        smiles = self.training_set[idx]
        tokenized = self.vocabulary.tokenize(smiles)
        encoded = self.vocabulary.encode(tokenized)
        if self.descriptors:
            descriptors = self.descriptors_training_set[idx]
        else:
            descriptors = None
        return smiles, encoded, descriptors

    def get_validation(self, n_smiles):
        n_smiles = min(n_smiles, len(self.validation_set))
        idx = np.random.choice(
            len(self.validation_set), n_smiles, replace=False
        )  # Random choices based on the indexes
        smiles = [self.validation_set[i] for i in idx]
        # smiles = np.random.choice(self.validation_set, n_smiles)
        tokenized = [self.vocabulary.tokenize(sm) for sm in smiles]
        encoded = [self.vocabulary.encode(tk) for tk in tokenized]
        if self.descriptors:
            descriptors = [self.descriptors_validation_set[i] for i in idx]
        else:
            descriptors = [None] * n_smiles
        return self.collate(list(zip(smiles, encoded, descriptors)))

    def __str__(self):
        return (
            "dataset containing "
            + str(len(self))
            + " SMILES with a vocabulary of "
            + str(len(self.vocabulary))
            + " characters"
        )


class SelfiesDataset(Dataset):
    """
    A dataset of chemical structures, provided in SELFIES format.
    """

    def __init__(
        self,
        selfies,
        max_len=None,
        vocab_file=None,
        training_split=0.9,
        collate_class=SmilesCollate,
    ):
        """
        Can be initiated from either a list of SELFIES, or a line-delimited
        file.

        Args:
            selfies (list): the complete set of SELFIES that constitute the
              training dataset
            selfies_file (string): line-delimited file containing the complete
              set of SELFIES that constitute the training dataset
            training_split (numeric): proportion of the dataset to withhold for
              validation loss calculation
        """

        # shuffle the SELFIES
        self.selfies = selfies
        np.random.shuffle(self.selfies)

        # create vocabulary or else read from file
        if vocab_file:
            self.vocabulary = SelfiesVocabulary(vocab_file=vocab_file)
        else:
            self.vocabulary = SelfiesVocabulary(selfies=self.selfies)

        # remove SMILES greater than max_len
        self.max_len = max_len
        if self.max_len is not None:
            self.selfies = [
                sf
                for sf in self.selfies
                if len(self.vocabulary.tokenize(sf)) <= self.max_len
            ]

        # split out a validation set
        n_selfies = len(self.selfies)
        border = int(n_selfies * training_split)
        self.training_set = self.selfies[:border]
        self.validation_set = self.selfies[border:]

        self.collate = collate_class(self.vocabulary)

    def __len__(self):
        return len(self.training_set)

    def __getitem__(self, idx):
        selfies = self.training_set[idx]
        tokenized = self.vocabulary.tokenize(selfies)
        encoded = self.vocabulary.encode(tokenized)
        return encoded

    def get_validation(self, n_selfies):
        selfies = np.random.choice(self.validation_set, n_selfies)
        tokenized = [self.vocabulary.tokenize(sf) for sf in selfies]
        encoded = [self.vocabulary.encode(tk) for tk in tokenized]
        return list(zip(selfies, encoded))

    def __str__(self):
        return (
            "dataset containing "
            + str(len(self))
            + " SELFIES with a vocabulary of "
            + str(len(self.vocabulary))
            + " characters"
        )


class Vocabulary:
    """
    Handles encoding and decoding of SMILES to and from one-hot vectors.
    """

    def __init__(self, smiles=None, smiles_file=None, vocab_file=None):
        """
        Can be initiated from either a list of SMILES, or a line-delimited
        SMILES file, or a file containing only tokens.

        Args:
            smiles (list): the complete set of SMILES that constitute the
              training dataset
            smiles_file (string): line-delimited file containing the complete
              set of SMILES that constitute the training dataset
            vocab_file (string): line-delimited file containing all tokens to
              be used in the vocabulary
        """
        if vocab_file is not None:
            # read tokens from file, and add to vocabulary
            self.characters = read_file(vocab_file)
        else:
            # read SMILES
            if smiles is not None:
                self.smiles = smiles
            elif smiles_file is not None:
                self.smiles = read_file(smiles_file)
            else:
                raise ValueError(
                    "must provide SMILES list or file to" + " instantiate Vocabulary"
                )
            # tokenize all SMILES in the input and add all tokens to vocabulary
            all_chars = [self.tokenize(sm) for sm in self.smiles]
            self.characters = np.unique(np.array(list(chain(*all_chars)))).tolist()

        # add padding token
        if "<PAD>" not in self.characters:
            # ... unless reading a padded vocabulary from file
            self.characters.append("<PAD>")

        # create dictionaries
        self.dictionary = {key: idx for idx, key in enumerate(self.characters)}
        self.reverse_dictionary = {value: key for key, value in self.dictionary.items()}

    """
    Regular expressions used to tokenize SMILES strings; borrowed from
    https://github.com/undeadpixel/reinvent-randomized/blob/master/models/vocabulary.py
    """
    REGEXPS = {
        "brackets": re.compile(r"(\[[^\]]*\])"),
        "2_ring_nums": re.compile(r"(%\d{2})"),
        "brcl": re.compile(r"(Br|Cl)"),
    }
    REGEXP_ORDER = ["brackets", "2_ring_nums", "brcl"]

    def tokenize(self, smiles):
        """
        Convert a SMILES string into a sequence of tokens.
        """

        def split_by(smiles, regexps):
            if not regexps:
                return list(smiles)
            regexp = self.REGEXPS[regexps[0]]
            splitted = regexp.split(smiles)
            tokens = []
            for i, split in enumerate(splitted):
                if i % 2 == 0:
                    tokens += split_by(split, regexps[1:])
                else:
                    tokens.append(split)
            return tokens

        tokens = split_by(smiles, self.REGEXP_ORDER)
        tokens = ["SOS"] + tokens + ["EOS"]
        return tokens

    def encode(self, tokens):
        """
        Encode a series of tokens into a (numeric) tensor.
        """
        vec = torch.zeros(len(tokens))
        for idx, token in enumerate(tokens):
            vec[idx] = self.dictionary[token]
        return vec.long()

    def decode(self, sequence):
        """
        Decode a series of tokens back to a SMILES.
        """
        chars = []
        for i in sequence:
            if i == self.dictionary["EOS"]:
                break
            if i != self.dictionary["SOS"]:
                chars.append(self.reverse_dictionary[i])
        smiles = "".join(chars)
        # smiles = smiles.replace("L", "Cl").replace("R", "Br")
        return smiles

    def write(self, output_file):
        """
        Write the list of tokens in a vocabulary to a line-delimited file.
        """
        with open(output_file, "w") as f:
            for char in self.characters:
                f.write(char + "\n")

    def __len__(self):
        return len(self.characters)

    def __str__(self):
        return (
            "vocabulary containing "
            + str(len(self))
            + " characters: "
            + format(self.characters)
        )


class SelfiesVocabulary:
    """
    Handles encoding and decoding of SELFIES to and from one-hot vectors.
    """

    def __init__(self, selfies=None, selfies_file=None, vocab_file=None):
        """
        Can be initiated from either a list of SELFIES, or a line-delimited
        SELFIES file.

        Args:
            selfies (list): the complete set of SELFIES that constitute the
              training dataset
            selfies_file (string): line-delimited file containing the complete
              set of SELFIES that constitute the training dataset
            vocab_file (string): line-delimited file containing all tokens to
              be used in the vocabulary
        """
        if vocab_file is not None:
            # read tokens from file, and add to vocabulary
            all_chars = read_file(vocab_file)
            # prevent chain popping open multi-character tokens
            self.characters = np.unique(
                np.array(list(chain(*[[char] for char in all_chars])))
            ).tolist()
        else:
            # read SMILES
            if selfies is not None:
                self.selfies = selfies
            elif selfies_file is not None:
                self.selfies = read_file(selfies_file)
            else:
                raise ValueError(
                    "must provide SELFIES list or file to" + " instantiate Vocabulary"
                )
            # tokenize all SMILES in the input and add all tokens to vocabulary
            alphabet = sorted(list(sf.get_alphabet_from_selfies(self.selfies)))
            self.characters = alphabet

            # add padding token
            self.characters.append("<PAD>")
            # add SOS/EOS tokens
            self.characters.append("SOS")
            self.characters.append("EOS")

        # create dictionaries
        self.dictionary = {key: idx for idx, key in enumerate(self.characters)}
        self.reverse_dictionary = {value: key for key, value in self.dictionary.items()}

    def tokenize(self, selfie):
        """
        Convert a SELFIES string into a sequence of tokens.
        """
        tokens = list(sf.split_selfies(selfie))
        tokens = ["SOS"] + tokens + ["EOS"]
        return tokens

    def encode(self, tokens):
        """
        Encode a series of tokens into a (numeric) tensor.
        """
        vec = torch.zeros(len(tokens))
        for idx, token in enumerate(tokens):
            vec[idx] = self.dictionary[token]
        return vec.long()

    def decode(self, sequence):
        """
        Decode a series of tokens back to a SELFIES.
        """
        chars = []
        for i in sequence:
            if i == self.dictionary["EOS"]:
                break
            if i != self.dictionary["SOS"]:
                chars.append(self.reverse_dictionary[i])
        smiles = "".join(chars)
        return smiles

    def write(self, output_file):
        """
        Write the list of tokens in a vocabulary to a line-delimited file.
        """
        with open(output_file, "w") as f:
            for char in self.characters:
                f.write(char + "\n")

    def __len__(self):
        return len(self.characters)

    def __str__(self):
        return (
            "vocabulary containing "
            + str(len(self))
            + " characters: "
            + format(self.characters)
        )


def vocabulary_from_representation(representation, smiles_or_selfies):
    if representation == "SMILES":
        return Vocabulary(smiles_or_selfies)
    elif representation == "SELFIES":
        return SelfiesVocabulary(smiles_or_selfies)
    else:
        raise ValueError(f"Unknown representation {representation}")


def Variable(tensor):
    """Wrapper for torch.autograd.Variable that also accepts
    numpy arrays directly and automatically assigns it to
    the GPU. Be aware in case some operations are better
    left to the CPU.
    Obtained from REINVENT source code."""
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if torch.cuda.is_available():
        return torch.autograd.Variable(tensor).cuda()
    return torch.autograd.Variable(tensor)
