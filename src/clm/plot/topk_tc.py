import argparse
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
import os
import numpy as np
from clm.functions import read_csv_file

parser = argparse.ArgumentParser(description=__doc__)


def add_args(parser):
    parser.add_argument(
        "--outcome_files",
        type=str,
        nargs="+",
        help="Paths of all the model evaluation files relevant to topk_tc ",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the save the output file",
    )

    return parser


def compute_topk_tc(outcome, min_tcs, ks):
    tc_count = {min_tc: [] for min_tc in min_tcs}
    n_total = len(outcome)

    for min_tc in min_tcs:
        for k in ks:
            n_rows_at_least_rank_k = outcome[
                (outcome["target_rank"] <= k) & (outcome["Tc"] >= min_tc)
            ]
            top_k_accuracy = (len(n_rows_at_least_rank_k) / n_total) * 100
            tc_count[min_tc].append(top_k_accuracy)

    return tc_count


def exact_tc_matches(outcome, min_tcs):
    match = []
    for min_tc in min_tcs:
        for smile, df in outcome.groupby("smiles"):
            # For each held out molecule, extracting tc of at least min_tc
            # We are assuming min_tc threshold to be an exact match in this case instead of inchikey
            matches = df[df["Tc"] >= min_tc]
            if matches.shape[0] > 0:
                match.append(matches.head(1).assign(min_tc=min_tc))
            else:
                # Assigning every other attribute as nan to avoid double representation of a single threshold
                match.append(
                    pd.DataFrame([{col: np.nan for col in df}]).assign(
                        min_tc=min_tc
                    )
                )

    return pd.concat(match)


def plot(outcome_files, output_dir):
    # Make output directory if it doesn't exist yet
    os.makedirs(output_dir, exist_ok=True)

    outcome = pd.concat(
        [
            read_csv_file(outcome_file, delimiter=",")
            for outcome_file in outcome_files
        ]
    )

    # Filtering out tc of more than 0.4 because that's considered to be an exact match for this case
    outcome = outcome[(outcome["target_source"] == "model")]
    min_tcs = [0.4, 0.675, 1]

    filtered_outcome = exact_tc_matches(outcome, min_tcs)

    ks = range(0, 30)
    tc_count = compute_topk_tc(filtered_outcome, min_tcs, ks)

    for min_tc in min_tcs:
        plt.step(ks, tc_count[min_tc], label=min_tc)

    plt.title(
        "Top-k accuracy curve when considering Tc >= 0.4, 0.675 as 'correct'"
    )
    plt.xlabel("k")
    plt.xscale("log")
    plt.ylabel("Top k %")
    plt.legend()

    file_name = Path(output_dir) / "topk_tc"
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
