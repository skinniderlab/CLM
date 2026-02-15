import logging
import os.path

import pandas as pd
import torch
from tqdm import tqdm

from clm.datasets import Vocabulary, SelfiesVocabulary
from clm.models import (
    RNN,
    ConditionalRNN,
    Transformer,
    StructuredStateSpaceSequenceModel,
    H3Model,
    H3ConvModel,
    HyenaModel,
)
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
        "--model_type",
        type=str,
        help="Type of model used (e.g., S4, Transformer)",
    )
    parser.add_argument(
        "--rnn_type", type=str, help="Type of RNN used (e.g., LSTM, GRU)"
    )
    parser.add_argument(
        "--embedding_size", type=int, help="Size of the embedding layer"
    )
    parser.add_argument(
        "--hidden_size", type=int, help="Size of the hidden layers"
    )
    parser.add_argument(
        "--n_layers", type=int, help="Number of layers in the model"
    )
    parser.add_argument(
        "--state_dim", type=int, help="State dimension for S4 model"
    )
    parser.add_argument(
        "--n_ssm", type=int, help="Number of SSM layers for S4 model"
    )
    parser.add_argument(
        "--n_heads", type=int, help="Number of heads for the model"
    )
    parser.add_argument(
        "--exp_factor", type=int, help="Expansion factor for Transformer model"
    )
    parser.add_argument(
        "--bias", action="store_true", help="Use bias in Transformer model"
    )
    parser.add_argument(
        "--use_fast_fftconv",
        action="store_true",
        help="Use fast FFT convolution for H3 or H3Conv model",
    )
    parser.add_argument(
        "--measure", type=str, help="Measure parameter for the H3 or H3Conv model"
    )
    parser.add_argument(
        "--mode", type=str, help="Mode parameter for the H3 or H3Conv model"
    )
    parser.add_argument(
        "--lr", type=float, help="Learning rate for the H3 or H3Conv model"
    )
    parser.add_argument("--order", type=int, help="Order for Hyena model")
    parser.add_argument(
        "--filter_order", type=int, help="Filter order for Hyena model"
    )
    parser.add_argument(
        "--dropout", type=float, help="Dropout rate for the RNN"
    )
    parser.add_argument(
        "--conditional",
        action="store_true",
        help="Activate Conditional RNN model",
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
    parser.add_argument(
        "--batch_size", type=int, help="Batch size for training"
    )
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
    model_type,
    rnn_type,
    embedding_size,
    hidden_size,
    n_layers,
    state_dim,
    n_ssm,
    n_heads,
    exp_factor,
    bias,
    use_fast_fftconv,
    measure,
    mode,
    lr,
    order,
    filter_order,
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

    if model_type == "S4":
        assert (
            heldout_file is not None
        ), "heldout_file must be provided for conditional RNN Model"
        heldout_dataset = load_dataset(
            representation=representation,
            input_file=heldout_file,
            vocab_file=vocab_file,
        )
        model = StructuredStateSpaceSequenceModel(
            vocabulary=vocab,  # heldout_dataset.vocabulary
            model_dim=embedding_size,
            state_dim=state_dim,
            n_layers=n_layers,
            n_ssm=n_ssm,
            dropout=dropout,
        )

    elif model_type == "H3":
        assert (
            heldout_file is not None
        ), "heldout_file must be provided for conditional RNN Model"
        heldout_dataset = load_dataset(
            representation=representation,
            input_file=heldout_file,
            vocab_file=vocab_file,
        )
        model = H3Model(
            vocabulary=vocab,
            n_layers=n_layers,
            model_dim=embedding_size,
            state_dim=state_dim,
            head_dim=n_heads,
            dropout=dropout,
            use_fast_fftconv=use_fast_fftconv,
            measure=measure,
            mode=mode,
            lr=lr,
        )

    elif model_type == "H3Conv":
        assert (
            heldout_file is not None
        ), "heldout_file must be provided for conditional RNN Model"
        heldout_dataset = load_dataset(
            representation=representation,
            input_file=heldout_file,
            vocab_file=vocab_file,
        )
        model = H3ConvModel(
            vocabulary=vocab,
            n_layers=n_layers,
            model_dim=embedding_size,
            head_dim=n_heads,
            dropout=dropout,
            use_fast_fftconv=use_fast_fftconv,
        )

    elif model_type == "Hyena":
        assert (
            heldout_file is not None
        ), "heldout_file must be provided for conditional RNN Model"
        heldout_dataset = load_dataset(
            representation=representation,
            input_file=heldout_file,
            vocab_file=vocab_file,
        )
        model = HyenaModel(
            vocabulary=vocab,
            n_layers=n_layers,
            d_model=embedding_size,
            order=order,
            filter_order=filter_order,
            num_heads=n_heads,
            dropout=dropout,
        )

    elif model_type == "Transformer":
        assert (
            heldout_file is not None
        ), "heldout_file must be provided for conditional RNN Model"
        heldout_dataset = load_dataset(
            representation=representation,
            input_file=heldout_file,
            vocab_file=vocab_file,
        )
        model = Transformer(
            vocabulary=vocab,
            n_blocks=n_layers,
            n_heads=n_heads,
            embedding_size=embedding_size,
            dropout=dropout,
            exp_factor=exp_factor,
            bias=bias,
        )

    elif model_type == "RNN":
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

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    logging.info(vocab.dictionary)

    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_file))
    else:
        model.load_state_dict(
            torch.load(model_file, map_location=torch.device("cpu"))
        )

    model.eval()

    # Erase file contents if there are any
    open(output_file, "w").close()

    with tqdm(total=sample_mols) as pbar:
        for i in range(0, sample_mols, batch_size):
            n_sequences = min(batch_size, sample_mols - i)
            descriptors = None
            if heldout_dataset is not None:
                # Use modulo to cycle through heldout_dataset
                descriptor_indices = [
                    (i + j) % len(heldout_dataset) for j in range(n_sequences)
                ]
                descriptors = torch.stack(
                    [heldout_dataset[idx][1] for idx in descriptor_indices]
                )
                descriptors = descriptors.to(model.device)
            sampled_smiles, losses = model.sample(
                descriptors=descriptors,
                n_sequences=n_sequences,
                return_losses=True,
            )
            df = pd.DataFrame(
                zip(losses, sampled_smiles), columns=["loss", "smiles"]
            )

            write_to_csv_file(
                output_file, mode="w" if i == 0 else "a+", info=df
            )

            pbar.update(batch_size)


def main(args):
    sample_molecules_RNN(
        representation=args.representation,
        model_type=args.model_type,
        rnn_type=args.rnn_type,
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        state_dim=args.state_dim,
        n_ssm=args.n_ssm,
        n_heads=args.n_heads,
        exp_factor=args.exp_factor,
        bias=args.bias,
        use_fast_fftconv=args.use_fast_fftconv,
        measure=args.measure,
        mode=args.mode,
        lr=args.lr,
        order=args.order,
        filter_order=args.filter_order,
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
