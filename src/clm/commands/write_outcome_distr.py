import argparse
import pandas as pd
from clm.functions import read_file

parser = argparse.ArgumentParser(description=__doc__)


def add_args(parser):
    parser.add_argument("--sample_file", type=str, help="Path to the sampled file")
    parser.add_argument("--train_file", type=str, help="Path to the train file")
    parser.add_argument("--max_mols", type=int, help="Number of samples to select")
    parser.add_argument("--pubchem_file", type=str, help="Path to the PubChem file")
    parser.add_argument(
        "--output_file", type=str, help="Path to the save the output file"
    )

    return parser


def write_outcome_distr(sample_file, max_mols, train_file, pubchem_file, output_file):

    sample_file = pd.read_csv(sample_file, delimiter=",")
    sample = sample_file.sample(
        n=max_mols, replace=True, weights=sample_file["size"], ignore_index=True
    )

    pubchem = pd.read_csv(
        pubchem_file, delimiter="\t", header=None, names=["smiles", "mass", "formula"]
    )
    pubchem = pubchem[pubchem["formula"].isin(set(sample.formula))]
    pubchem = pubchem.drop_duplicates(subset=["formula"], keep="first")

    train = pd.DataFrame({"smiles": read_file(train_file, smile_only=True)})
    combination = pd.concat(
        [
            sample.assign(source="model"),
            pubchem.assign(source="pubchem"),
            train.assign(source="train"),
        ]
    )
    combination.to_csv(output_file, index=False, columns=["smiles", "source"])


def main(args):
    write_outcome_distr(
        sample_file=args.sample_file,
        max_mols=args.max_mols,
        train_file=args.train_file,
        pubchem_file=args.pubchem_file,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)