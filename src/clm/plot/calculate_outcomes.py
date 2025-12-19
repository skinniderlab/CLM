import argparse
import pandas as pd
from pathlib import Path
import os
from matplotlib import pyplot as plt
import seaborn as sns
from clm.functions import read_csv_file


def add_args(parser):
    parser.add_argument(
        "--outcome_files",
        type=str,
        nargs="+",
        help="Paths of all the model evaluation files relevant to calculate outcomes ",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to directory to save resulting images(s) at",
    )
    return parser


def plot(outcome_files, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    # Concatenate all the outcome files
    outcome = pd.concat(
        [read_csv_file(file, delimiter=",") for file in outcome_files]
    )

    # Plot every figure possible
    for outcome_type, _outcome in outcome.groupby("outcome"):
        _data = {}
        for bin, df in _outcome.groupby("bin"):
            if bin != "all":
                _data[bin] = df["value"].tolist()

        # rearrange by bin ascending
        data = []
        labels = []
        sorted_bins = sorted(_data.keys(), key=lambda x: int(x.split("-")[0]))
        for bin in sorted_bins:
            labels.append(bin)
            data.append(_data[bin])

        sns.violinplot(
            data=data,
            palette=sns.color_palette("pastel"),
            linecolor="auto",
            width=0.4,
            inner="box",
            inner_kws={"box_width": 3},
            native_scale=True,
        )

        outcome_type = str(outcome_type)
        plt.title(outcome_type)
        plt.ylabel(outcome_type)
        plt.xlabel("Frequency")
        plt.xticks(list(range(len(labels))), labels)

        file_name = Path(output_dir) / outcome_type
        plt.savefig(file_name)

        # Clear the previous figure
        plt.clf()


def main(args):
    plot(
        outcome_files=args.outcome_files,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
