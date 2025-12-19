import pandas as pd
import numpy as np
from clm.functions import write_to_csv_file, read_csv_file


def add_args(parser):
    parser.add_argument(
        "--input_file", type=str, nargs="+", help="Path to the input CSV file."
    )
    parser.add_argument(
        "--cv_file", type=str, nargs="+", help="Path to the  training sets."
    )
    parser.add_argument(
        "--output_file", type=str, help="Path to the output CSV file."
    )
    parser.add_argument(
        "--summary_fn",
        type=str,
        default="freq_avg",
        help="Summary function (fp10k/freq_sum/freq_avg).",
    )
    parser.add_argument(
        "--min_freq",
        type=int,
        default=1,
        help="Minimum frequency of molecules to consider.",
    )
    return parser


def process_tabulated_molecules(
    input_file, cv_files, output_file, summary_fn, min_freq=1
):
    meta = pd.concat(
        [
            read_csv_file(file, dtype={"smiles": str, "inchikey": str}).assign(
                fold=idx
            )
            for idx, file in enumerate(input_file)
        ]
    )

    data = meta.pivot_table(
        index="inchikey",
        columns="fold",
        values="size",
        aggfunc="first",
        fill_value=0,
    )

    # Filter out molecules with frequency less than min_freq
    data = data[data.sum(axis=1) >= min_freq]

    uniq_inchikeys = data.index.to_numpy()

    for fold_idx, cv_file in enumerate(cv_files):
        cv_dat = read_csv_file(cv_file, usecols=["inchikey"])
        cv_dat = cv_dat[cv_dat["inchikey"].isin(uniq_inchikeys)]

        if not cv_dat.empty:
            data.loc[cv_dat["inchikey"], fold_idx] = np.nan

    # Optionally normalize by total sampling frequency
    if summary_fn == "fp10k":
        data = 10e3 * data / np.nansum(data, axis=0)

    # Calculate mean/sum
    if summary_fn == "freq-sum":
        # With what frequency (across all folds)
        # were valid molecules produced by our models?
        data = pd.DataFrame(
            {"inchikey": list(uniq_inchikeys), "size": np.nansum(data, axis=1)}
        )
    else:
        # With what average frequency (across all folds)
        # were valid molecules produced by our models?
        data = pd.DataFrame(
            {"inchikey": list(uniq_inchikeys), "size": np.nanmean(data, axis=1)}
        )

    data = data.sort_values(by="size", ascending=False, kind="stable").query(
        "size > 0"
    )

    if not data.empty:
        # Add metadata (mass and formula)
        data = data.merge(
            meta.drop_duplicates("inchikey")[
                ["inchikey", "smiles", "mass", "formula"]
            ],
            how="left",
            on="inchikey",
        )

    write_to_csv_file(output_file, data)


def main(args):
    process_tabulated_molecules(
        input_file=args.input_file,
        cv_files=args.cv_file,
        output_file=args.output_file,
        summary_fn=args.summary_fn,
        min_freq=args.min_freq,
    )
