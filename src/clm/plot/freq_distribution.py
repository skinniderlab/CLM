import argparse
from pathlib import Path
from collections import Counter
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os
import logging
import numpy as np
import matplotlib.cbook as cbook
from clm.functions import read_csv_file

logger = logging.getLogger(__name__)


def add_args(parser):
    parser.add_argument(
        "--outcome_files",
        type=str,
        nargs="+",
        help="Paths of all the model evaluation files relevant to frequency distribution ",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to directory to save resulting images(s) at",
    )
    return parser


def plot_distribution(novel_outcomes, output_dir):
    logger.info("Plotting frequency distribution")
    data = list(novel_outcomes[True])

    counter = Counter(data)
    x, y = counter.keys(), counter.values()

    sns.scatterplot(x=x, y=y)
    plt.title("Frequency Distribution Plot")
    plt.xlabel("Sampling Frequency")
    plt.ylabel("# of unique molecules")

    # Plot logarithmic scale
    plt.yscale("log")
    plt.xscale("log")

    file_path = Path(output_dir) / "freq_distr_scatter"
    plt.savefig(file_path)

    # Clear the previous figure
    plt.clf()


def plot_box_plot(novel_outcomes, output_dir):
    logger.info("Plotting frequency box plot")
    novel_molecules_count = np.array(novel_outcomes.get(True, []))
    known_molecules_count = np.array(novel_outcomes.get(False, []))

    # The `CategoricalPlotter` object in `sns.boxplot` makes a copy of the
    # entire underlying DataFrame! (see the `comp_data` property). This is
    # extremely slow and memory-intensive. So we follow a manual approach ala
    # https://stackoverflow.com/questions/29895754
    bxpstats = []
    if len(novel_molecules_count):
        bxpstats.extend(
            cbook.boxplot_stats(
                np.ravel(novel_molecules_count), labels=["Novel molecules"]
            )
        )
    if len(known_molecules_count):
        bxpstats.extend(
            cbook.boxplot_stats(
                np.ravel(known_molecules_count), labels=["Known molecules"]
            )
        )

    fig, ax = plt.subplots(1, 1)
    ax.bxp(bxpstats)
    ax.set_yscale("log")

    # Direct but very slow/memory-intensive
    # sns.boxplot(data=(novel_molecules_count, known_molecules_count), width=0.2, log_scale=(False, True))

    plt.title(
        "Frequency with which test set molecules sampled from a given CV fold"
    )
    plt.ylabel("Sampling Frequency")

    file_path = Path(output_dir) / "freq_distr_box"
    plt.savefig(file_path)

    # Clear the previous figure
    plt.clf()


def plot(outcome_files, output_dir):
    # Make output directory if it doesn't exist yet
    os.makedirs(output_dir, exist_ok=True)

    columns = []
    outcome = []

    for outcome_file in outcome_files:
        logger.info(f"Reading outcome file {outcome_file}")
        data = read_csv_file(
            outcome_file, delimiter=",", usecols=("is_novel", "size")
        )
        logger.info(f"Read outcome dataframe of shape {data.shape}")
        columns = data.columns.values.tolist()
        data = data.values.tolist()
        outcome.extend(data)

    logger.info("Creating combined DataFrame")
    outcome = pd.DataFrame(outcome, columns=columns)
    logger.info(f"Created combined DataFrame of shape {outcome.shape}")

    # Dictionary of novelty (True or False) mapped to list of frequency of generation
    # E.g. True : [1, 1, 4, 5] means there are four distinct novel smiles
    # each of which were generated  once, once, four, and five times
    novel_outcomes = {
        is_novel: df["size"].tolist()
        for is_novel, df in outcome.groupby("is_novel")
    }

    plot_distribution(novel_outcomes, output_dir)
    plot_box_plot(novel_outcomes, output_dir)


def main(args):
    plot(
        outcome_files=args.outcome_files,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
