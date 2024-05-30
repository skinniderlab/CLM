from clm.plot.calculate_outcomes import plot as calculate_outcomes
from clm.plot.write_nn_tc import plot as write_nn_tc
from clm.plot.train_discriminator import plot as train_discriminator
from clm.plot.freq_distribution import plot as freq_distribution
from clm.plot.nn_tc_ever_v_never import plot as nn_tc_ever_v_never
from clm.plot.calculate_outcome_distrs import plot as calculate_outcome_distrs
from clm.plot.topk_tc import plot as topk_tc
from clm.plot.topk import plot as topk


def add_args(parser):
    parser.add_argument(
        "evaluation_type",
        type=str,
        help="Type of evaluation you want figures of. Valid options are:  \n"
        " calculate_outcomes, write_nn_tc, train_discriminator, freq_distribution, topk_tc \n",
    )
    parser.add_argument(
        "--outcome_files",
        type=str,
        nargs="+",
        help="Paths of all the model evaluation files relevant to a specific plot",
    )
    parser.add_argument(
        "--ranks_file",
        type=str,
        required=False,
        help="Path to the rank file ",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to directory to save resulting images(s) at",
    )
    return parser


def plot(evaluation_type, outcome_files, output_dir, ranks_file=None):
    if evaluation_type == "calculate_outcomes":
        calculate_outcomes(outcome_files, output_dir)
    elif evaluation_type == "write_nn_tc":
        write_nn_tc(outcome_files, output_dir)
    elif evaluation_type == "train_discriminator":
        train_discriminator(outcome_files, output_dir)
    elif evaluation_type == "freq_distribution":
        freq_distribution(outcome_files, output_dir)
    elif evaluation_type == "nn_tc_ever_v_never":
        nn_tc_ever_v_never(outcome_files, ranks_file, output_dir)
    elif evaluation_type == "calculate_outcome_distrs":
        calculate_outcome_distrs(outcome_files, output_dir)
    elif evaluation_type == "topk_tc":
        topk_tc(outcome_files, output_dir)
    elif evaluation_type == "topk":
        topk(outcome_files, output_dir)


def main(args):
    plot(
        evaluation_type=args.evaluation_type,
        outcome_files=args.outcome_files,
        output_dir=args.output_dir,
        ranks_file=args.ranks_file,
    )
