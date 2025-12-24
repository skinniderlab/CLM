"""
Datasets used by PyTorch for language modelling of chemical structures.
"""

import numpy as np
import re
import selfies as sf
import torch
from torch.nn.utils.rnn import pad_sequence
from itertools import chain
import pandas as pd
from torch.utils.data import Dataset
from clm.functions import read_file


class SmilesDataset(Dataset):
    """
    A dataset of chemical structures, provided in SMILES format.
    """

    def __init__(self, data, max_len=None, vocab_file=None, training_split=0.9):
        """
        Can be initiated from either a list of SMILES, or a line-delimited
        file.

        Args:
            data (pd.Dataframe): A Dataframe with (at least) a "smiles"
              column, that constitutes the training dataset
            vocab_file (string): line-delimited file containing all tokens to
              be used in the vocabulary
            training_split (numeric): proportion of the dataset to withhold for
              validation loss calculation
        """
        # Maintain indices to the data
        self.indices = np.arange(len(data))

        # create vocabulary or else read from file
        if vocab_file:
            self.vocabulary = Vocabulary(vocab_file=vocab_file)
        else:
            self.vocabulary = Vocabulary(smiles=data["smiles"].tolist())

        # remove SMILES greater than max_len
        self.max_len = max_len
        if self.max_len is not None:
            valid_indices = data.apply(
                lambda row: len(self.vocabulary.tokenize(row["smiles"]))
                <= self.max_len,
                axis=1,
            )
            self.indices = self.indices[valid_indices]

        # shuffle filtered indices, and create the `.data` Dataframe
        np.random.shuffle(self.indices)
        self.data = data.iloc[self.indices].reset_index(drop=True)

        # maintain a list of descriptor columns that we have available
        # (any numeric column that is not 'smiles' or 'inchikey')
        # These are possibly useful property values to return on __getitem__
        self.descriptor_names = [
            c
            for c in self.data.select_dtypes(include=np.number).columns
            if c not in ("smiles", "inchikey")
        ]
        self.n_descriptors = len(self.descriptor_names)

        # convert dtypes of descriptors to floats
        for desc in self.descriptor_names:
            self.data[desc] = self.data[desc].astype(float)

        # split out a validation set
        n_smiles = len(self.data)
        border = int(n_smiles * training_split)
        self.training_set = self.data[:border]
        self.validation_set = self.data[border:]

        # define collate function
        self.collate = SmilesCollate(self.vocabulary)

    def __len__(self):
        return len(self.training_set)

    def __getitem__(self, idx):
        row = self.training_set.iloc[idx]
        tokenized = self.vocabulary.tokenize(row["smiles"])
        try:
            encoded = self.vocabulary.encode(tokenized)
        except (
            KeyError
        ):  # some callers might only be interested in the descriptors
            encoded = None
        descriptors = torch.Tensor(
            pd.to_numeric(row[self.descriptor_names]).to_numpy()
        )
        return encoded, descriptors    

    def get_validation(self, n_smiles):
        selected_indices = np.random.choice(self.validation_set.index, n_smiles)
        selected_data = self.validation_set.loc[selected_indices]
        smiles = selected_data["smiles"]
        descriptors = torch.Tensor(
            selected_data[self.descriptor_names].to_numpy()
        )
        tokenized = [self.vocabulary.tokenize(sm) for sm in smiles]
        encoded = [self.vocabulary.encode(tk) for tk in tokenized]
        return self.collate(list(zip(encoded, descriptors)))

    def __str__(self):
        return (
            "dataset containing "
            + str(len(self))
            + " SMILES with a vocabulary of "
            + str(len(self.vocabulary))
            + " characters"
        )


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

    def __init__(self, vocabulary):
        self.padding_token = vocabulary.dictionary["<PAD>"]

    def __call__(self, item):
        encoded, descriptors = zip(*item)
        padded = pad_sequence(encoded, padding_value=self.padding_token)
        lengths = [len(seq) for seq in encoded]
        return padded, lengths, torch.stack(descriptors)


class SelfiesDataset(Dataset):
    """
    A dataset of chemical structures, provided in SELFIES format.
    """

    def __init__(self, data, max_len=None, vocab_file=None, training_split=0.9):
        """
        Can be initiated from either a list of SELFIES, or a line-delimited
        file.

        Args:
            data (pd.Dataframe): A Dataframe with (at least) a "smiles"
              column, that constitutes the training dataset
            training_split (numeric): proportion of the dataset to withhold for
              validation loss calculation
        """

        # shuffle the SELFIES
        self.selfies = data["smiles"]
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

        # define collate function
        self.collate = SmilesCollate(self.vocabulary)

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
        return self.collate(encoded)

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
            self.characters = read_file(vocab_file)["smiles"].to_list()
        else:
            # read SMILES
            if smiles is not None:
                self.smiles = smiles
            elif smiles_file is not None:
                self.smiles = read_file(smiles_file)
            else:
                raise ValueError(
                    "must provide SMILES list or file to"
                    + " instantiate Vocabulary"
                )
            # tokenize all SMILES in the input and add all tokens to vocabulary
            all_chars = [self.tokenize(sm) for sm in self.smiles]
            self.characters = np.unique(
                np.array(list(chain(*all_chars)))
            ).tolist()

        # add padding token
        if "<PAD>" not in self.characters:
            # ... unless reading a padded vocabulary from file
            self.characters.append("<PAD>")

        # create dictionaries
        self.dictionary = {key: idx for idx, key in enumerate(self.characters)}
        self.reverse_dictionary = {
            value: key for key, value in self.dictionary.items()
        }

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
                    "must provide SELFIES list or file to"
                    + " instantiate Vocabulary"
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
        self.reverse_dictionary = {
            value: key for key, value in self.dictionary.items()
        }

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
