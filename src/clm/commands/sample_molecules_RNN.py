import logging
import os.path

import pandas as pd
import torch
from tqdm import tqdm

from clm.datasets import Vocabulary, SelfiesVocabulary
from clm.models import RNN, ConditionalRNN
from clm.functions import write_to_csv_file, read_csv_file

logger = logging.getLogger(__name__)


def add_args(parser):
    parser.add_argument(
        "--representation",
        type=str,
        default="SMILES",
        help="Molecular representation format (one of: SMILES/SELFIES)",
    )

    parser.add_argument(
        "--rnn_type", type=str, help="Type of RNN used (e.g., LSTM, GRU)"
    )

    parser.add_argument(
        "--embedding_size", type=int, help="Size of the embedding layer"
    )

    parser.add_argument("--hidden_size", type=int, help="Size of the hidden layers")

    parser.add_argument("--n_layers", type=int, help="Number of layers in the RNN")

    parser.add_argument("--dropout", type=float, help="Dropout rate for the RNN")

    parser.add_argument("--batch_size", type=int, help="Batch size for training")

    parser.add_argument(
        "--sample_mols", type=int, help="Number of molecules to generate"
    )

    parser.add_argument(
        "--vocab_file",
        type=str,
        required=True,
        help="Output path for the vocabulary file ({fold} is populated automatically)",
    )

    parser.add_argument(
        "--model_file", type=str, help="File path to the saved trained model"
    )
    parser.add_argument(
        "--output_file", type=str, help="File path to save the output file"
    )
    parser.add_argument(
        "--conditional_rnn",
        action="store_true",
    )
    parser.add_argument(
        "--minmax_descriptor_file",
        type=str,
        default=None,
        help="File path for storing min and max of all the descriptors which would be responsible as inputs for sampling",
    )
    parser.add_argument(
        "--sample_descriptor_file",
        type=str,
        default=None,
        help="File path for sample descriptors",
    )

    return parser


def sample_molecules_RNN(
    representation,
    rnn_type,
    embedding_size,
    hidden_size,
    n_layers,
    dropout,
    batch_size,
    sample_mols,
    vocab_file,
    model_file,
    output_file,
    conditional_rnn=False,
    minmax_descriptor_file=None,
    sample_descriptor_file=None,
):
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    logging.info(f"cuda: {torch.cuda.is_available()}")

    if representation == "SELFIES":
        vocab = SelfiesVocabulary(vocab_file=vocab_file)
    else:
        vocab = Vocabulary(vocab_file=vocab_file)

    descriptors = None

    if conditional_rnn:
        if sample_descriptor_file is None:
            descriptors_csv = read_csv_file(minmax_descriptor_file)
            min_vals = descriptors_csv["min_val"].values
            max_vals = descriptors_csv["max_val"].values

            descriptors = torch.rand((sample_mols, len(min_vals))) * (
                torch.tensor(max_vals) - torch.tensor(min_vals)
            ) + torch.tensor(min_vals)
        else:
            descriptor_input = read_csv_file(sample_descriptor_file, delimiter=",")
            # Handle inchikey and smiles if present drop the columns
            columns_dropped = ["smiles", "inchikey"]
            descriptor_input = descriptor_input.drop(
                columns=[
                    col for col in columns_dropped if col in descriptor_input.columns
                ]
            )

            # Handle the order of columns from a min_max_descriptor_file
            descriptors_csv = read_csv_file(minmax_descriptor_file)
            descriptor_col = descriptors_csv["descriptor"].values
            descriptor_input = descriptor_input[descriptor_col]
            descriptors = descriptor_input.values

            descriptors = torch.tensor(descriptors, dtype=torch.float32)

        model = ConditionalRNN(
            vocab,
            rnn_type=rnn_type,
            n_layers=n_layers,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            dropout=dropout,
            num_descriptors=descriptors.shape[1],
        )

    else:
        model = RNN(
            vocab,
            rnn_type=rnn_type,
            n_layers=n_layers,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            dropout=dropout,
        )
    logging.info(vocab.dictionary)

    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_file))
    else:
        model.load_state_dict(torch.load(model_file, map_location=torch.device("cpu")))

    model.eval()

    # Erase file contents if there are any
    open(output_file, "w").close()

    with tqdm(total=sample_mols) as pbar:
        for i in range(0, sample_mols, batch_size):
            end_idx = i + min(batch_size, sample_mols - i)
            if conditional_rnn:
                batch_descriptors = descriptors[i:end_idx, :]
                sampled_smiles, losses = model.sample(
                    batch_descriptors, return_losses=True
                )
            else:
                sampled_smiles, losses = model.sample(end_idx - i, return_losses=True)
            df = pd.DataFrame(zip(losses, sampled_smiles), columns=["loss", "smiles"])
            write_to_csv_file(output_file, mode="w" if i == 0 else "a+", info=df)
            pbar.update(batch_size)


def main(args):
    sample_molecules_RNN(
        representation=args.representation,
        rnn_type=args.rnn_type,
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        sample_mols=args.sample_mols,
        vocab_file=args.vocab_file,
        model_file=args.model_file,
        output_file=args.output_file,
        conditional_rnn=args.conditional_rnn,
        minmax_descriptor_file=args.minmax_descriptor_file,
        sample_descriptor_file=args.sample_descriptor_file,
    )
