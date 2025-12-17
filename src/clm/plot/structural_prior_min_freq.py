import argparse
from pathlib import Path
import os
import re
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from clm.plot.topk import topk
from clm.functions import read_csv_file


def add_args(parser):
    parser.add_argument(
        "--tc_files",
        type=str,
        nargs="+",
        help="Path to ranks file ",
    )
    parser.add_argument(
        "--rank_files",
        type=str,
        nargs="+",
        help="Path to ranks file ",
    )
    parser.add_argument(
        "--sampled_files",
        type=str,
        nargs="+",
        help="Path to sampled file ",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        help="Path to test file ",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to directory to save resulting images(s) at",
    )
    return parser


def plot_topk(min_freqs, rank_files, output_dir):
    outcomes = []
    for filename, freq in zip(rank_files, min_freqs.keys()):
        outcomes.append(
            read_csv_file(filename, delimiter=",", index_col=0).assign(min_freqs=freq)
        )
    outcomes = pd.concat(outcomes)
    outcomes = outcomes[outcomes["target_source"] == "model"]

    topk(
        outcomes,
        list(min_freqs.keys()),
        output_dir,
        target_column="min_freqs",
        title=f"Top-k accuracy for structures sampled at least {list(min_freqs.keys())} times",
        filename="top_k_min_freq",
    )


def plot_tc_freq(min_freqs, tc_files, output_dir):
    for filename, freq in zip(tc_files, min_freqs.keys()):
        outcome = read_csv_file(filename, delimiter=",", index_col=0)
        sampled_df = outcome[outcome["target_source"] == "model"]
        top_rank = sampled_df[sampled_df["target_rank"] == 0]
        min_freqs[freq] = list(top_rank["Tc"])

    sns.violinplot(
        data=min_freqs,
        palette=sns.color_palette("pastel"),
        linecolor="auto",
        width=0.4,
        inner="box",
        inner_kws={"box_width": 3},
        native_scale=True,
    )

    plt.title("Tc - for structures sampled at least 1, 2, 3 and 4 times")
    plt.ylabel("Tanimoto coefficient")
    plt.xlabel("Minimum frequency")

    file_name = Path(output_dir) / "tc_min_freq"
    plt.savefig(file_name)

    # Clear the previous figure
    plt.clf()


def plot_p_accuracy(min_freqs, rank_files, output_dir):
    for filename, freq in zip(rank_files, min_freqs.keys()):
        outcome = read_csv_file(filename, delimiter=",", index_col=0)
        sampled_df = outcome[outcome["target_source"] == "model"]
        accurate = sampled_df[sampled_df["target_rank"] == 0]
        min_freqs[freq] = round((len(accurate) / len(sampled_df)) * 100, 2)

    plt.bar(list(min_freqs.keys()), list(min_freqs.values()))
    plt.xlabel("Minimum frequency")
    plt.ylabel("Accuracy (%)")

    file_name = Path(output_dir) / "p_accuracy"
    plt.savefig(file_name)

    # Clear the previous figure
    plt.clf()


def plot_p_ever_generated(
    min_freqs,
    rank_files,
    output_dir,
):
    for filename, freq in zip(rank_files, min_freqs.keys()):
        outcome = read_csv_file(filename, delimiter=",")
        outcome = outcome[outcome["target_source"] == "model"]
        ever_generated = outcome[outcome["target_rank"].notnull()]
        min_freqs[freq] = round((len(ever_generated) / len(outcome)) * 100, 2)

    plt.bar(list(min_freqs.keys()), list(min_freqs.values()))
    plt.xlabel("Minimum frequency")
    plt.ylabel("% ever generated")

    file_name = Path(output_dir) / "p_ever_generated"
    plt.savefig(file_name)

    # Clear the previous figure
    plt.clf()


def plot(rank_files, tc_files, output_dir):
    # Make an output directory if it doesn't yet
    os.makedirs(output_dir, exist_ok=True)

    # Extract minimum frequency from file names
    rank_freqs = {
        re.search(r"(\d)", os.path.basename(filename)).group(1): None
        for filename in rank_files
    }
    tc_freqs = {
        re.search(r"(\d)", os.path.basename(filename)).group(1): None
        for filename in tc_files
    }

    plot_tc_freq(tc_freqs, tc_files, output_dir)
    plot_p_accuracy(rank_freqs, rank_files, output_dir)
    plot_p_ever_generated(rank_freqs, rank_files, output_dir)
    plot_topk(rank_freqs, rank_files, output_dir)


def main(args):
    plot(
        rank_files=args.rank_files,
        tc_files=args.tc_files,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
