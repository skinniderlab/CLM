import argparse
from pathlib import Path
import pandas as pd
import seaborn as sns
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from clm.functions import read_csv_file


def add_args(parser):
    parser.add_argument(
        "--outcome_files",
        type=str,
        nargs="+",
        help="Paths of all the model evaluation files relevant to train discriminator",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to directory to save resulting images(s) at",
    )
    return parser


def plot_roc_curve(outcome, output_dir):
    y_test, y_scores = list(outcome["y"].dropna()), list(
        outcome["y_prob_1"].dropna()
    )

    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve, classifying training vs. generated molecules")
    plt.legend(loc="lower right")

    filepath = Path(output_dir) / "train_disc_line"
    plt.savefig(filepath)

    # Clear the previous figure
    plt.clf()


def plot_confusion_matrix(outcome, output_dir):
    y_test, y_pred = list(outcome["y"].dropna()), list(
        outcome["y_pred"].dropna()
    )

    cm = confusion_matrix(y_test, y_pred)
    # Normalize the confusion matrix to show percentages
    cm_percentage = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    classes = ["Known", "Generated"]
    # Plot confusion matrix
    sns.heatmap(
        cm_percentage,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Confusion Matrix, classifying training vs. generated molecules")

    filepath = Path(output_dir) / "train_disc_confusion_matrix"
    plt.savefig(filepath)

    # Clear the previous figure
    plt.clf()


def plot(outcome_files, output_dir):
    # Make output directory if it doesn't exist yet
    os.makedirs(output_dir, exist_ok=True)

    outcome = pd.concat(
        [
            read_csv_file(outcome_file, delimiter=",")
            for outcome_file in outcome_files
        ]
    )

    # Metrics for outcomes (from `sklearn.metrics`)
    # acc = accuracy_score(outcome['y_test'], outcome['y_pred'])
    # auc = roc_auc_score(outcome['y_test'], outcome['y_probs'])
    # auprc = average_precision_score(outcome['y_test'], outcome['y_probs'])

    plot_roc_curve(outcome, output_dir)
    plot_confusion_matrix(outcome, output_dir)


def main(args):
    plot(
        outcome_files=args.outcome_files,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
