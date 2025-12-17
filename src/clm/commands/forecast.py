import numpy as np
import os
import pandas as pd
from scipy.stats import chisquare
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
)
import logging
from rdkit import rdBase
from clm.functions import read_csv_file

# suppress rdkit errors
rdBase.DisableLog("rdApp.error")

logger = logging.getLogger(__name__)


def add_args(parser):
    parser.add_argument(
        "--test_file", type=str, required=True, help="File path of test file"
    )
    parser.add_argument(
        "--sample_file", type=str, required=True, help="File path of sample file"
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="File path of output file"
    )
    parser.add_argument(
        "--max-mols",
        type=int,
        help="Maximum number of molecules to read from sample file",
    )
    return parser


def forecast(test_file, sample_file, output_file, max_molecules=None):
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    test = read_csv_file(test_file)
    deepmet = read_csv_file(sample_file, nrows=max_molecules)
    deepmet = deepmet.assign(known=deepmet["inchikey"].isin(test["inchikey"]))
    deepmet = deepmet.sort_values(by=["size"], ascending=False, kind="stable")

    # extract ROC/PR curves
    y = deepmet["known"].tolist()
    x = deepmet["size"].tolist()
    x_rnd = deepmet.sample(frac=1)["size"].tolist()
    fpr, tpr, thresholds = roc_curve(y, x)
    roc_df = pd.DataFrame(
        {
            "mode": "true",
            "curve": "ROC",
            "fpr": fpr,
            "tpr": tpr,
            "threshold": thresholds,
        }
    )
    precision, recall, thresholds = precision_recall_curve(y, x)
    pr_df = pd.DataFrame(
        {
            "mode": "true",
            "curve": "PR",
            "precision": precision,
            "recall": recall,
            "threshold": np.append(thresholds, np.nan),
        }
    )

    # calculate shuffled ROC/PR curves
    fpr, tpr, thresholds = roc_curve(y, x_rnd)
    roc_df_rnd = pd.DataFrame(
        {
            "mode": "random",
            "curve": "ROC",
            "fpr": fpr,
            "tpr": tpr,
            "threshold": thresholds,
        }
    )
    precision, recall, thresholds = precision_recall_curve(y, x_rnd)
    pr_df_rnd = pd.DataFrame(
        {
            "mode": "random",
            "curve": "PR",
            "precision": precision,
            "recall": recall,
            "threshold": np.append(thresholds, np.nan),
        }
    )

    try:
        auc = roc_auc_score(y, x)
        auc_rnd = roc_auc_score(y, x_rnd)
    except ValueError:
        logger.warning(
            "Only one class present in y_true. ROC AUC score is not defined in that case."
        )
        auc = None
        auc_rnd = None

    auprc = average_precision_score(y, x)
    auprc_rnd = average_precision_score(y, x_rnd)

    ranks = []
    step_sizes, current_step = [], 10
    deepmet_max = deepmet.shape[0]

    while current_step <= 1000000:
        step_sizes.append(current_step)
        current_step *= 10

    for step in step_sizes:
        ranks.extend(range(step, min(step * 10, deepmet_max), step))

    if deepmet_max > 1000000:
        ranks.extend(range(1000000, deepmet_max, 1000000))
    ranks.append(deepmet_max)

    ranks = sorted(set(ranks))

    ef = pd.DataFrame({"rank": ranks, "EF": np.nan, "n_known": 0, "pval": np.nan})
    for idx, rank in enumerate(ranks):
        obs = sum(y[0:rank])
        exp = rank * sum(y) / deepmet.shape[0]
        # exp will be 0 when every single generated SMILE is known
        try:
            EF = obs / exp
        except ZeroDivisionError as e:
            logger.warning(f"Cannot compute enrichment factor. {e}")
            EF = None

        # chi-square test
        f_obs = np.array([obs, rank - obs]) / rank
        f_exp = np.array([exp, rank - exp]) / rank
        pval = chisquare(f_obs=f_obs, f_exp=f_exp).pvalue

        ef.at[idx, "EF"] = EF
        ef.at[idx, "n_known"] = obs
        ef.at[idx, "pval"] = pval

    output_df = pd.concat([roc_df, pr_df, roc_df_rnd, pr_df_rnd])
    score_df = pd.DataFrame(
        {
            "score": ["auroc", "auprc"] * 2,
            "mode": ["true"] * 2 + ["random"] * 2,
            "value": [auc, auprc, auc_rnd, auprc_rnd],
        }
    )
    output_df = output_df._append(score_df)
    output_df = output_df._append(ef)

    output_df.to_csv(output_file, index=False)


def main(args):
    forecast(
        test_file=args.test_file,
        sample_file=args.sample_file,
        output_file=args.output_file,
        max_molecules=args.max_mols,
    )
