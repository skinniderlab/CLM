import argparse
import logging
import os.path
import torch
from tqdm import tqdm

from clm.datasets import Vocabulary, SelfiesVocabulary
from clm.models import RNN
from clm.functions import set_seed, seed_type

logger = logging.getLogger(__name__)


def add_args(parser):
    parser.add_argument(
        "--representation",
        type=str,
        default="SMILES",
        help="Molecular representation format (one of: SMILES/SELFIES)",
    )

    parser.add_argument(
        "--seed",
        type=seed_type,
        default=None,
        nargs="?",
        help="Random seed for reproducibility",
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

    parser.add_argument("--sample_mols", type=int, help="Number of molecules to generate")

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

    return parser


def sample_molecules_RNN(
    representation,
    seed,
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
):
    set_seed(seed)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    logging.info(f"cuda: {torch.cuda.is_available()}")

    if representation == "SELFIES":
        vocab = SelfiesVocabulary(vocab_file=vocab_file)
    else:
        vocab = Vocabulary(vocab_file=vocab_file)

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
        for _ in range(0, sample_mols, batch_size):
            sampled_smiles, losses = model.sample(batch_size, return_losses=True)

            with open(output_file, "a+") as f:
                for loss, sm in zip(losses, sampled_smiles):
                    f.write(f"{round(loss, 4)}, {sm} \n")

            pbar.update(batch_size)


def main(args):
    sample_molecules_RNN(
        representation=args.representation,
        seed=args.seed,
        rnn_type=args.rnn_type,
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        sample_mols=args.sample_mols,
        vocab_file=args.vocab_file,
        model_file=args.model_file,
        output_file=args.output_file
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
