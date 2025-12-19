import logging
from clm.functions import read_file, write_to_csv_file, read_csv_file


logger = logging.getLogger(__name__)


def add_args(parser):
    parser.add_argument(
        "--sampled_file", type=str, help="Path to the sampled file"
    )
    parser.add_argument("--test_file", type=str, help="Path to the test file ")
    parser.add_argument(
        "--output_file", type=str, help="Path to the save the output file"
    )

    return parser


def write_freq_distribution(sampled_file, test_file, output_file):
    logger.info(f"Reading sampled data from {sampled_file}")
    sampled_data = read_csv_file(sampled_file)
    logger.info(f"Reading test smiles from {test_file}")
    test_smiles = set(
        read_file(test_file, stream=False, smile_only=True)["smiles"]
    )

    # Label smiles not found in test set as novel
    logger.info("Finding novel smiles in sampled data")
    sampled_data["is_novel"] = True
    sampled_data.loc[sampled_data["smiles"].isin(test_smiles), "is_novel"] = (
        False
    )

    # Store values of is_novel column as true or false instead of 0 or 1
    sampled_data["is_novel"] = sampled_data["is_novel"].astype(bool)

    write_to_csv_file(output_file, sampled_data)
    smile_distribution = sampled_data.reset_index(drop=True)
    return smile_distribution


def main(args):
    write_freq_distribution(
        sampled_file=args.sampled_file,
        test_file=args.test_file,
        output_file=args.output_file,
    )
