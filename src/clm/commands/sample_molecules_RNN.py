import logging
import os.path

import pandas as pd
import torch
from tqdm import tqdm

from clm.datasets import Vocabulary, SelfiesVocabulary
from clm.models import RNN, ConditionalRNN
from clm.functions import load_dataset, write_to_csv_file

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

    parser.add_argument(
        "--conditional", action="store_true", help="Activate Conditional RNN model"
    )
    parser.add_argument(
        "--conditional_emb",
        action="store_true",
        help="Add descriptor with the input smiles without passing it through an embedding layer",
    )
    parser.add_argument(
        "--conditional_emb_l",
        action="store_true",
        help="Pass the descriptors through an embedding layer and add descriptor with the input smiles",
    )
    parser.add_argument(
        "--conditional_dec",
        action="store_true",
        help="Add descriptor with the rnn output without passing it through decoder layer",
    )
    parser.add_argument(
        "--conditional_dec_l",
        action="store_true",
        help="Pass the descriptors through a decoder layer and add descriptor with the rnn output",
    )
    parser.add_argument(
        "--conditional_h",
        action="store_true",
        help="Add descriptor in hidden and cell state",
    )
    parser.add_argument(
        "--heldout_file",
        type=str,
        nargs="+",
        help="Testing file in heldout set. Useful for sampling from a Conditional RNN model",
    )

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
    conditional=False,
    conditional_emb=False,
    conditional_emb_l=True,
    conditional_dec=False,
    conditional_dec_l=True,
    conditional_h=False,
    heldout_file=None,
):
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    logging.info(f"cuda: {torch.cuda.is_available()}")

    if representation == "SELFIES":
        vocab = SelfiesVocabulary(vocab_file=vocab_file)
    else:
        vocab = Vocabulary(vocab_file=vocab_file)

    heldout_dataset = None
    if conditional:
        assert (
            heldout_file is not None
        ), "heldout_file must be provided for conditional RNN Model"
        heldout_dataset = load_dataset(
            representation=representation,
            input_file=heldout_file,
            vocab_file=vocab_file,
        )
        model = ConditionalRNN(
            vocab,
            rnn_type=rnn_type,
            n_layers=n_layers,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            dropout=dropout,
            num_descriptors=heldout_dataset.n_descriptors,
            conditional_emb=conditional_emb,
            conditional_emb_l=conditional_emb_l,
            conditional_dec=conditional_dec,
            conditional_dec_l=conditional_dec_l,
            conditional_h=conditional_h,
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
            n_sequences = min(batch_size, sample_mols - i)
            descriptors = None
            if heldout_dataset is not None:
                max_idx = len(heldout_dataset)
                n_sequences = min(n_sequences, max_idx - i)
                if n_sequences <= 0:
                    break
                descriptors = torch.stack(
                    [heldout_dataset[_i][1] for _i in range(i, i + n_sequences)]
                ).to(model.device)

            sampled_smiles, losses = model.sample(
                descriptors=descriptors, n_sequences=n_sequences, return_losses=True
            )
            df_dict = {"loss": losses, "smiles": sampled_smiles}
            if descriptors is not None:
                df_dict["descriptors"] = descriptors.cpu().numpy().tolist()
            df = pd.DataFrame(df_dict)

            write_to_csv_file(output_file, mode="w" if i == 0 else "a+", info=df)
            pbar.update(n_sequences)


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
        conditional=args.conditional,
        conditional_emb=args.conditional_emb,
        conditional_emb_l=args.conditional_emb_l,
        conditional_dec=args.conditional_dec,
        conditional_dec_l=args.conditional_dec_l,
        conditional_h=args.conditional_h,
        heldout_file=args.heldout_file,
    )
