import logging
import os
import os.path
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from rdkit import rdBase
from clm.models import (
    RNN,
    ConditionalRNN,
    Transformer,
    StructuredStateSpaceSequenceModel,
    H3Model,
    H3ConvModel,
    HyenaModel,
)
from clm.loggers import EarlyStopping, track_loss, print_update
from clm.functions import write_smiles, load_dataset

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# suppress Chem.MolFromSmiles error output
rdBase.DisableLog("rdApp.error")
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
        "--bias", type=bool, help="Use bias in Transformer model"
    )
    parser.add_argument(
        "--use_fast_fftconv",
        type=bool,
        help="Use fast FFT convolution for H3 or H3Conv model",
    )
    parser.add_argument("--order", type=int, help="Order for Hyena model")
    parser.add_argument(
        "--filter_order", type=int, help="Filter order for Hyena model"
    )
    parser.add_argument(
        "--inner_factor", type=int, help="Inner factor for Hyena model"
    )
    parser.add_argument(
        "--dropout", type=float, help="Dropout rate for the RNN"
    )
    parser.add_argument(
        "--batch_size", type=int, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, help="Learning rate for the optimizer"
    )
    parser.add_argument(
        "--max_epochs", type=int, help="Maximum number of epochs for training"
    )
    parser.add_argument(
        "--patience", type=int, help="Patience for early stopping"
    )
    parser.add_argument(
        "--log_every_steps", type=int, help="Logging frequency in steps"
    )
    parser.add_argument(
        "--log_every_epochs", type=int, help="Logging frequency in epochs"
    )
    parser.add_argument(
        "--sample_mols",
        type=int,
        help="Number of molecules to sample for evaluation",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input file path for training data",
    )
    parser.add_argument(
        "--vocab_file",
        type=str,
        required=True,
        help="Output path for the vocabulary file ({fold} is populated automatically)",
    )
    parser.add_argument(
        "--smiles_file",
        type=str,
        default=None,
        help="File path for additional SMILES data (optional)",
    )
    parser.add_argument(
        "--model_file", type=str, help="File path to save the trained model"
    )
    parser.add_argument(
        "--loss_file", type=str, help="File path to save the training loss data"
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
    return parser


def training_step(batch, model, optim, dataset, batch_size):
    loss = model.loss(batch)
    optim.zero_grad()
    loss.backward()
    optim.step()
    validation = dataset.get_validation(batch_size)
    validation_loss = model.loss(validation)
    return loss, validation_loss


def sample_and_write_smiles(
    model, sample_mols, batch_size, smiles_file, dataset=None
):
    sampled_smiles = []
    with tqdm(total=sample_mols) as pbar:
        while len(sampled_smiles) < sample_mols:
            descriptors = None
            if dataset is not None:
                _, _, descriptors = dataset.get_validation(batch_size)
                descriptors = descriptors.to(model.device)
            new_smiles = model.sample(
                n_sequences=batch_size,
                return_smiles=True,
                descriptors=descriptors,
            )
            sampled_smiles.extend(new_smiles)
            pbar.update(len(new_smiles))
    write_smiles(sampled_smiles, smiles_file)


def train_models_RNN(
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
    order,
    filter_order,
    inner_factor,
    dropout,
    batch_size,
    learning_rate,
    max_epochs,
    patience,
    log_every_steps,
    log_every_epochs,
    sample_mols,
    input_file,
    vocab_file,
    smiles_file,
    model_file,
    loss_file,
    conditional=False,
    conditional_emb=False,
    conditional_emb_l=True,
    conditional_dec=False,
    conditional_dec_l=True,
    conditional_h=False,
):

    os.makedirs(os.path.dirname(os.path.abspath(model_file)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(loss_file)), exist_ok=True)

    dataset = load_dataset(representation, input_file, vocab_file)

    if model_type == "S4":
        model = StructuredStateSpaceSequenceModel(
            vocabulary=dataset.vocabulary,
            model_dim=embedding_size,
            state_dim=state_dim,
            n_layers=n_layers,
            n_ssm=n_ssm,
            dropout=dropout,
        )

    elif model_type == "H3":
        model = H3Model(
            vocabulary=dataset.vocabulary,
            n_layers=n_layers,
            d_model=embedding_size,
            d_state=state_dim,
            head_dim=n_heads,
            dropout=dropout,
            use_fast_fftconv=use_fast_fftconv,
        )

    elif model_type == "H3Conv":
        model = H3ConvModel(
            vocabulary=dataset.vocabulary,
            n_layers=n_layers,
            d_model=embedding_size,
            head_dim=n_heads,
            dropout=dropout,
            use_fast_fftconv=use_fast_fftconv,
        )

    elif model_type == "Hyena":
        model = HyenaModel(
            vocabulary=dataset.vocabulary,
            n_layers=n_layers,
            d_model=embedding_size,
            order=order,
            filter_order=filter_order,
            num_heads=n_heads,
            dropout=dropout,
            inner_factor=inner_factor,
        )

    elif model_type == "Transformer":
        model = Transformer(
            vocabulary=dataset.vocabulary,
            n_blocks=n_layers,
            n_heads=n_heads,
            embedding_size=embedding_size,
            dropout=dropout,
            exp_factor=exp_factor,
            bias=bias,
        )

    elif model_type == "RNN":
        if conditional:
            model = ConditionalRNN(
                dataset.vocabulary,
                rnn_type=rnn_type,
                n_layers=n_layers,
                embedding_size=embedding_size,
                hidden_size=hidden_size,
                dropout=dropout,
                num_descriptors=dataset.n_descriptors,
                conditional_emb=conditional_emb,
                conditional_emb_l=conditional_emb_l,
                conditional_dec=conditional_dec,
                conditional_dec_l=conditional_dec_l,
                conditional_h=conditional_h,
            )
        else:
            model = RNN(
                dataset.vocabulary,
                rnn_type=rnn_type,
                n_layers=n_layers,
                embedding_size=embedding_size,
                hidden_size=hidden_size,
                dropout=dropout,
            )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    logger.info(dataset.vocabulary.dictionary)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate
    )
    optim = Adam(
        model.parameters(), betas=(0.9, 0.999), eps=1e-08, lr=learning_rate
    )
    early_stop = EarlyStopping(patience=patience)

    for epoch in range(max_epochs):
        for batch_no, batch in tqdm(enumerate(loader), total=len(loader)):
            loss, validation_loss = training_step(
                batch, model, optim, dataset, batch_size
            )

            loop_count = (epoch * len(loader)) + batch_no + 1
            if loop_count % log_every_steps == 0 or (
                log_every_epochs and (batch_no + 1) == len(loader)
            ):
                track_loss(
                    loss_file,
                    epoch + 1,
                    loop_count,
                    value=[loss.item(), validation_loss.item()],
                )

                print_update(
                    model,
                    epoch,
                    batch_no + 1,
                    loss.item(),
                    validation_loss.item(),
                    dataset=dataset,
                )

            early_stop(validation_loss.item(), model, model_file, loop_count)

            if early_stop.stop:
                logging.info("Early stopping triggered.")
                break

        if early_stop.stop:
            break

    if log_every_epochs or log_every_steps:
        track_loss(
            loss_file,
            epoch=[None],
            batch_no=[early_stop.step_at_best],
            value=[early_stop.best_loss],
            outcome=["training loss"],
        )

    model.load_state_dict(torch.load(model_file))
    model.eval()
    if smiles_file:
        sample_and_write_smiles(
            model, sample_mols, batch_size, smiles_file, dataset
        )


def main(args):
    train_models_RNN(
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
        order=args.order,
        filter_order=args.filter_order,
        inner_factor=args.inner_factor,
        dropout=args.dropout,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        patience=args.patience,
        log_every_steps=args.log_every_steps,
        log_every_epochs=args.log_every_epochs,
        sample_mols=args.sample_mols,
        input_file=args.input_file,
        vocab_file=args.vocab_file,
        smiles_file=args.smiles_file,
        model_file=args.model_file,
        loss_file=args.loss_file,
        conditional=args.conditional,
        conditional_emb=args.conditional_emb,
        conditional_emb_l=args.conditional_emb_l,
        conditional_dec=args.conditional_dec,
        conditional_dec_l=args.conditional_dec_l,
        conditional_h=args.conditional_h,
    )
