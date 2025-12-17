import argparse
import pandas as pd
from pathlib import Path
import os
from matplotlib import pyplot as plt
import seaborn as sns
from clm.functions import read_csv_file, split_frequency_ranges


def add_args(parser):
    parser.add_argument(
        "--outcome_files",
        type=str,
        nargs="+",
        help="Paths of all the model evaluation files relevant to write_nn_tc",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to directory to save resulting images(s) at",
    )
    return parser


def plot_generated_v_ref(outcome, output_dir):
    source_map = {
        "DeepMet": "Generated metabolites",
        "PubChem": "Negative control (PubChem)",
    }

    data, labels = [], []
    for source, df in outcome.groupby("source"):
        data.append(df["nn_tc"].tolist())
        labels.append(source_map[source])

    sns.violinplot(
        data=data,
        palette=sns.color_palette("pastel"),
        linecolor="auto",
        width=0.2,
        inner="box",
        inner_kws={"box_width": 3},
        native_scale=True,
    )

    plt.title("nearest-neighbor Tc, generated vs. PubChem")
    plt.ylabel("Nearest-neighbor Tc")
    plt.xticks(list(range(len(labels))), labels)

    file_name = Path(output_dir) / "nn_tc_gen_v_pubchem"
    plt.savefig(file_name)

    # Clear the previous figure
    plt.clf()


def plot_by_frequency(outcome, output_dir):
    # Split the outcomes by frequency bins
    outcome_freq = split_frequency_ranges(outcome, max_molecules=10000000)

    _data = {}
    for bin, df in outcome_freq.groupby("bin"):
        _data[bin] = df["nn_tc"].tolist()

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

    plt.title("nearest-neighbor Tc, by frequency")
    plt.ylabel("Nearest-neighbor Tc")
    plt.xlabel("Frequency")
    plt.xticks(list(range(len(labels))), labels)

    file_name = Path(output_dir) / "nn_tc_by_freq"
    plt.savefig(file_name)

    # Clear the previous figure
    plt.clf()


def plot(outcome_files, output_dir):
    # Make an output directory if it doesn't yet
    os.makedirs(output_dir, exist_ok=True)

    # Concatenate all the outcome files
    outcome = pd.concat([read_csv_file(file, delimiter=",") for file in outcome_files])

    plot_generated_v_ref(outcome, output_dir)
    plot_by_frequency(outcome, output_dir)


def main(args):
    plot(outcome_files=args.outcome_files, output_dir=args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
