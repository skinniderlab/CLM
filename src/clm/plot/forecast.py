import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import os
from sklearn.metrics import auc
from matplotlib import pyplot as plt
import seaborn as sns
from clm.functions import read_csv_file

logger = logging.getLogger(__name__)


def add_args(parser):
    parser.add_argument(
        "--outcome_files",
        type=str,
        help="Paths of all the model evaluation files relevant to nn_tc_ever_v_never ",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to directory to save resulting images(s) at",
    )
    return parser


def plot_roc_curve(outcome, output_dir):
    outcome = outcome[outcome["curve"] == "ROC"]

    plt.figure()
    for mode in ("true", "random"):
        _outcome = outcome[outcome["mode"] == mode]
        tpr, fpr = _outcome["tpr"], _outcome["fpr"]
        try:
            roc_auc = auc(fpr, tpr)
        except ValueError as e:
            logger.warning(f" Couldn't generate ROC curve. {e}")
            return
        plt.plot(
            fpr,
            tpr,
            lw=2,
            label=f"{mode} ROC curve (area = {roc_auc:.2})",
        )
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve showing recovery of test set molecules")
    plt.legend(loc="lower right")

    filepath = Path(output_dir) / "forecast_roc"
    plt.savefig(filepath)

    # Clear the previous figure
    plt.clf()


def plot_distribution(outcome, output_dir):
    logger.info("Plotting frequency distribution")

    enrichment_factor = outcome["EF"].dropna().tolist()
    rank = outcome["rank"].dropna().tolist()
    p_value = outcome["pval"].dropna().tolist()

    # For cases when either of these values are empty
    # This only happens when all generated SMILES are novel
    if not (enrichment_factor and rank and p_value):
        logger.warning("Not enough values to plot the Enrichment Factor Curve")
        return

    neg_log_p_value = [-np.log10(p) for p in p_value]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    sns.scatterplot(x=rank, y=neg_log_p_value, ax=axes[0])
    axes[0].set_title("Fold enrichment for test-set molecules in model output")
    axes[0].set_xlabel("# of top-ranked molecules")
    axes[0].set_ylabel(r"$-\log_{10}(\mathrm{P})$")
    axes[0].set_xscale("log")

    sns.scatterplot(x=rank, y=enrichment_factor, ax=axes[1])
    axes[1].set_xlabel("# of top-ranked molecules")
    axes[1].set_ylabel("Fold enrichment")
    axes[1].set_xscale("log")

    file_path = Path(output_dir) / "fold_enrichment.png"
    plt.savefig(file_path)
    plt.close(fig)


def plot(outcome_files, output_dir):
    # Make an output directory if it doesn't exist yet
    os.makedirs(output_dir, exist_ok=True)

    outcome = pd.concat(
        [
            read_csv_file(outcome_file, delimiter=",")
            for outcome_file in outcome_files
        ]
    )

    plot_roc_curve(outcome, output_dir)
    plot_distribution(outcome, output_dir)


def main(args):
    plot(
        outcome_files=args.outcome_files,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
