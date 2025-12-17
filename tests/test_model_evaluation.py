from pathlib import Path
import pandas as pd
from clm.commands.calculate_outcomes import (
    get_dataframes,
    calculate_outcomes_dataframe,
    calculate_outcomes,
    prep_outcomes_freq,
)
from clm.commands.write_nn_Tc import prep_nn_tc, write_nn_Tc
from clm.commands.train_discriminator import train_discriminator
from clm.commands.write_freq_distribution import write_freq_distribution
from clm.commands.calculate_outcome_distrs import (
    calculate_outcome_distr,
    write_outcome_distr,
)
from clm.commands.add_carbon import add_carbon
from clm.commands.plot import plot
from clm.plot.topk_tc import exact_tc_matches
from clm.functions import (
    assert_checksum_equals,
    read_csv_file,
    local_seed,
    set_seed,
)

base_dir = Path(__file__).parent.parent
test_dir = base_dir / "tests/test_data"


def test_generate_outcome_dicts():
    prep_sample_df = read_csv_file(test_dir / "prep_outcomes_freq.csv")
    train_df, sample_df = get_dataframes(
        train_file=test_dir
        / "snakemake_output/0/prior/inputs/train0_LOTUS_truncated_SMILES_0.smi",
        prep_sample_df=prep_sample_df,
    )

    # Certain fields in train_dict/gen_dict need to be tolerant of Nones
    train_df["qed"].iloc[4] = None
    sample_df["SA"].iloc[3] = None

    calculate_outcomes_dataframe(sample_df, train_df)


def test_prep_outcome_freq(tmp_path):
    with local_seed(12):
        outcomes = prep_outcomes_freq(
            samples=test_dir
            / "snakemake_output/0/prior/samples/LOTUS_truncated_SMILES_0_unique_masses.csv",
            max_molecules=500,
            known_smiles=test_dir
            / "snakemake_output/0/prior/samples/known_LOTUS_truncated_SMILES_0_unique_masses.csv",
            invalid_smiles=test_dir
            / "snakemake_output/0/prior/samples/invalid_LOTUS_truncated_SMILES_0_unique_masses.csv",
        )

        true_outcomes = read_csv_file(test_dir / "prep_outcomes_freq.csv")
        pd.testing.assert_frame_equal(
            outcomes, true_outcomes, check_dtype=False
        )


def test_calculate_outcomes(tmp_path):
    set_seed(12)
    output_file = tmp_path / "calculate_outcome.csv"
    outcomes = calculate_outcomes(
        sampled_file=test_dir
        / "snakemake_output/0/prior/samples/LOTUS_truncated_SMILES_0_unique_masses.csv",
        max_molecules=500,
        known_smiles=test_dir
        / "snakemake_output/0/prior/samples/known_LOTUS_truncated_SMILES_0_unique_masses.csv",
        invalid_smiles=test_dir
        / "snakemake_output/0/prior/samples/invalid_LOTUS_truncated_SMILES_0_unique_masses.csv",
        # For LOTUS, train/test "_all.smi" files are the same
        train_file=test_dir / "test_LOTUS_SMILES_all_trunc.smi",
        output_file=output_file,
    )

    # % unique for bin "1-1" (if present) should be 1.0 (since all molecules are unique)
    unique_1 = outcomes[
        (outcomes.bin == "1-1") & (outcomes.outcome == "% unique")
    ]["value"].values
    if len(unique_1) > 0:
        assert unique_1[0] == 1.0

    # % unique for bin "2-2" (if present) should be 0.5 (since all molecules are generated twice)
    unique_2 = outcomes[
        (outcomes.bin == "2-2") & (outcomes.outcome == "% unique")
    ]["value"].values
    if len(unique_2) > 0:
        assert unique_2[0] == 0.5

    true_outcomes = read_csv_file(
        test_dir / "calculate_outcome.csv", keep_default_na=True
    )

    # https://stackoverflow.com/questions/14224172
    pd.testing.assert_frame_equal(
        outcomes.sort_values(["outcome", "bin"]).reset_index(drop=True),
        true_outcomes.sort_values(["outcome", "bin"]).reset_index(drop=True),
        check_like=False,
    )

    plot(
        evaluation_type="calculate_outcomes",
        outcome_files=[output_file],
        output_dir=tmp_path,
    )


def test_prep_nn_tc(tmp_path):
    with local_seed(0):
        outcomes = prep_nn_tc(
            sample_file=test_dir / "prep_nn_tc_input.csv",
            max_molecules=100,
            pubchem_file=test_dir / "PubChem_truncated.tsv",
        )

        true_outcomes = read_csv_file(test_dir / "prep_nn_tc_output.csv")
        pd.testing.assert_frame_equal(
            outcomes.sort_index(axis=1)
            .sort_values(["smiles", "formula"])
            .reset_index(drop=True),
            true_outcomes.sort_index(axis=1)
            .sort_values(["smiles", "formula"])
            .reset_index(drop=True),
        )


def test_write_nn_tc(tmp_path):
    query_file = test_dir / "input_write_nn_tc_query_file.csv"
    reference_file = test_dir / "input_write_nn_tc_reference_file.csv"
    output_file = tmp_path / "output_write_nn_tc.csv"
    outcomes = write_nn_Tc(
        query_file=query_file,
        reference_file=reference_file,
        pubchem_file=test_dir / "input_write_nn_tc_pubchem.tsv",
        max_molecules=100,
        output_file=output_file,
    )
    true_outcomes = read_csv_file(test_dir / "output_write_nn_tc.csv")
    pd.testing.assert_frame_equal(
        outcomes.reset_index(drop=True),
        true_outcomes.reset_index(drop=True),
        check_like=True,
    )

    plot(
        evaluation_type="write_nn_tc",
        outcome_files=[output_file],
        output_dir=tmp_path,
    )


def test_write_freq_distribution(tmp_path):
    output_file = tmp_path / "write_freq_distribution.csv"
    outcomes = write_freq_distribution(
        sampled_file=test_dir / "LOTUS_SMILES_processed_freq-avg_trunc.csv",
        test_file=test_dir / "test_LOTUS_SMILES_all_trunc.smi",
        output_file=output_file,
    )

    true_outcomes = read_csv_file(test_dir / "write_freq_distribution.csv")
    pd.testing.assert_frame_equal(outcomes, true_outcomes)

    plot(
        evaluation_type="freq_distribution",
        outcome_files=[output_file],
        output_dir=tmp_path,
    )


def test_train_discriminator(tmp_path):
    output_file = tmp_path / "train_discriminator.csv"
    outcomes = train_discriminator(
        train_file=test_dir
        / "snakemake_output/0/prior/inputs/train0_LOTUS_truncated_SMILES_0.smi",
        sample_file=test_dir
        / "snakemake_output/0/prior/samples/LOTUS_truncated_SMILES_processed_freq-avg.csv",
        max_mols=50000,
        output_file=output_file,
        seed=0,
    )

    true_outcomes = read_csv_file(test_dir / "train_discriminator.csv")
    pd.testing.assert_frame_equal(outcomes, true_outcomes)

    plot(
        evaluation_type="train_discriminator",
        outcome_files=[output_file],
        output_dir=tmp_path,
    )


def test_write_outcome_distr(tmp_path):
    outcomes = write_outcome_distr(
        sample_file=test_dir
        / "snakemake_output/0/prior/samples/LOTUS_truncated_SMILES_0_unique_masses.csv",
        max_mols=50,
        train_file=test_dir
        / "snakemake_output/0/prior/inputs/train_LOTUS_truncated_SMILES_0.smi",
        pubchem_file=test_dir / "PubChem_truncated.tsv",
    )

    true_outcomes = read_csv_file(test_dir / "write_outcome_distr.csv")
    pd.testing.assert_frame_equal(
        outcomes.reset_index(drop=True), true_outcomes.reset_index(drop=True)
    )


def test_outcome_distr(tmp_path):
    output_file = tmp_path / "outcome_distr.csv"

    outcomes = calculate_outcome_distr(
        sample_file=test_dir
        / "snakemake_output/0/prior/samples/LOTUS_truncated_SMILES_0_unique_masses.csv",
        max_mols=40,
        train_file=test_dir
        / "snakemake_output/0/prior/inputs/train_LOTUS_truncated_SMILES_0.smi",
        pubchem_file=test_dir / "PubChem_truncated.tsv",
        output_file=output_file,
    )

    true_outcomes = read_csv_file(test_dir / "outcome_distr.csv")
    pd.testing.assert_frame_equal(outcomes, true_outcomes)


def test_add_carbon(tmp_path):
    add_carbon(
        input_file=test_dir
        / "snakemake_output/0/prior/inputs/train0_LOTUS_truncated_SMILES_0.smi",
        output_file=tmp_path / "add_carbon.csv",
    )

    assert_checksum_equals(
        tmp_path / "add_carbon.csv", test_dir / "add_carbon.csv"
    )
    assert_checksum_equals(
        tmp_path / "add_carbon-unique.smi", test_dir / "add_carbon-unique.smi"
    )


def test_nn_tc_ever_never(tmp_path):
    query_file = (
        test_dir
        / "snakemake_output/0/prior/inputs/train0_LOTUS_truncated_SMILES_0.smi"
    )
    reference_file = (
        test_dir
        / "snakemake_output/0/prior/inputs/train0_LOTUS_truncated_SMILES_0.smi"
    )
    output_file = tmp_path / "output_nn_tc_ever_never.csv"
    write_nn_Tc(
        query_file=query_file,
        reference_file=reference_file,
        output_file=output_file,
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(output_file),
        pd.read_csv(test_dir / "output_nn_tc_ever_never.csv"),
    )


def test_tc_matches(tmp_path):
    data = {
        "smiles": [
            "CONCCNOOOC",
            "CCNOOCONNO",
            "CONONNCNON",
            "COCNOONOON",
            "ONONNCNNNC",
            "CCCNNCONON",
            "OOCCNCNCOO",
            "OCONCCNCON",
            "NCOCNNCCNO",
            "NOCCOOONNO",
        ],
        "Tc": [
            0.8048256115805752,
            0.547905510270021,
            0.20606892168056123,
            0.10223785371617233,
            0.5084079285260388,
            0.4671584918857752,
            0.057206875573343474,
            0.189574167585153,
            0.3260046265902604,
            0.2328257013952244,
        ],
    }
    df = pd.DataFrame.from_dict(data)
    min_tcs = [0.1, 0.4, 0.5]
    outcomes = exact_tc_matches(df, min_tcs).reset_index(drop=True)

    true_outcomes = read_csv_file(test_dir / "min_tc_match_output.csv")
    pd.testing.assert_frame_equal(outcomes, true_outcomes)
