from pathlib import Path
import pandas as pd

from clm.commands import (
    preprocess,
    create_training_sets,
    train_models_RNN,
    sample_molecules_RNN,
    tabulate_molecules,
    collect_tabulated_molecules,
    process_tabulated_molecules,
    write_structural_prior_CV,
    write_formula_prior_CV,
    plot,
)
import pytest
from clm.functions import assert_checksum_equals, read_csv_file, set_seed

base_dir = Path(__file__).parent.parent

test_dir = base_dir / "tests/test_data/snakemake_output"
dataset = base_dir / "tests/test_data/LOTUS_truncated.txt"
pubchem_tsv_file = base_dir / "tests/test_data/PubChem_truncated.tsv"


@pytest.mark.parametrize(
    (
        "input_file",
        "output_file",
        "max_input_smiles",
        "min_heavy_atoms",
        "neutralise",
        "valid_atoms",
    ),
    [
        (
            dataset,
            test_dir / "prior/raw/LOTUS_truncated.txt",
            1000,
            3,
            True,
            ["Br", "C", "Cl", "F", "H", "I", "N", "O", "P", "S"],
        ),
        (
            base_dir / "tests/test_data/smiles.csv",
            base_dir / "tests/test_data/smiles.csv",
            None,
            0,
            False,
            ["Br", "C", "Cl", "F", "I", "N", "O", "P", "S", "Se", "Si"],
        ),
    ],
)
def test_00_preprocess(
    tmp_path,
    input_file,
    output_file,
    max_input_smiles,
    min_heavy_atoms,
    neutralise,
    valid_atoms,
):
    preprocess.preprocess(
        input_file=input_file,
        output_file=tmp_path / "preprocessed.smi",
        max_input_smiles=max_input_smiles,
        min_heavy_atoms=min_heavy_atoms,
        neutralise=neutralise,
        valid_atoms=valid_atoms,
    )
    generated = (
        pd.read_csv(tmp_path / "preprocessed.smi")[["smiles", "inchikey"]]
        .sort_values("inchikey")
        .reset_index(drop=True)
    )
    oracle = (
        pd.read_csv(output_file)[["smiles", "inchikey"]]
        .sort_values("inchikey")
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(generated, oracle)


def test_00_preprocess_no_filter(tmp_path):
    # All SMILES in this file are canonicalized
    # So, under lax conditions, no SMILES should be excluded during the preprocessing step
    input_file = base_dir / "tests/test_data/smiles.csv"
    preprocess.preprocess(
        input_file=base_dir / "tests/test_data/smiles.csv",
        output_file=tmp_path / "preprocessed.smi",
        max_input_smiles=0,
        min_heavy_atoms=0,
        neutralise=False,
        # These are the unique elements found in the molecules listed in the input file
        valid_atoms=["Br", "C", "Cl", "F", "I", "N", "O", "P", "S", "Se", "Si"],
    )
    generated = (
        pd.read_csv(tmp_path / "preprocessed.smi")[["smiles", "inchikey"]]
        .sort_values("inchikey")
        .reset_index(drop=True)
    )
    # We are testing the input file against the original one to ensure no SMILES are inadvertently filtered out
    oracle = (
        pd.read_csv(input_file)[["smiles", "inchikey"]]
        .sort_values("inchikey")
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(generated, oracle)


def test_00_preprocess_keep_duplicates(tmp_path):
    # All SMILES in this file are valid but have duplications
    # Under relaxed conditions, and when the keep_duplicates
    # flag is set to true, no SMILES should be excluded during the preprocessing step
    input_file = base_dir / "tests/test_data/clean_smiles.csv"
    preprocess.preprocess(
        input_file=input_file,
        output_file=tmp_path / "preprocessed.smi",
        max_input_smiles=0,
        min_heavy_atoms=0,
        neutralise=False,
        # These are the unique elements found in the molecules listed in the input file
        valid_atoms=["Br", "C", "Cl", "F", "I", "N", "O", "P", "S", "Se", "Si"],
        keep_duplicates=True,
    )
    generated = (
        pd.read_csv(tmp_path / "preprocessed.smi")[["smiles", "inchikey"]]
        .sort_values("inchikey")
        .reset_index(drop=True)
    )
    # We are testing the input file against the original one to ensure no SMILES are inadvertently filtered out
    oracle = (
        pd.read_csv(input_file)[["smiles", "inchikey"]]
        .sort_values("inchikey")
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(generated, oracle)


def test_01_create_training_sets(tmp_path):
    folds = 3
    for fold in range(folds):
        set_seed(5831)
        create_training_sets.create_training_sets(
            input_file=test_dir / "prior/raw/LOTUS_truncated.txt",
            train0_file=tmp_path / "train0_file_{fold}",
            train_file=tmp_path / "train_file_{fold}",
            vocab_file=tmp_path / "vocabulary_file_{fold}",
            test0_file=tmp_path / "test0_file_{fold}",
            enum_factor=0,
            folds=folds,
            which_fold=fold,
            representation="SMILES",
            min_tc=0,
            max_input_smiles=1000,
        )
    # `train0_file_0` denotes the train smiles without augmentation for fold
    # 0; Since we're running with enum_factor=0, this should be identical
    # to `train_file_0` (train smiles with augmentation for fold 0)
    assert_checksum_equals(
        tmp_path / "train0_file_0", tmp_path / "train_file_0"
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(tmp_path / "train_file_0"),
        pd.read_csv(
            test_dir / "0/prior/inputs/train_LOTUS_truncated_SMILES_0.smi"
        ),
    )
    assert_checksum_equals(
        tmp_path / "vocabulary_file_0",
        test_dir / "0/prior/inputs/train_LOTUS_truncated_SMILES_0.vocabulary",
    )

    # Check that the same InChI key does not appear in both the training and test set in any CV split
    test0_all = []
    for fold in range(folds):
        train0 = read_csv_file(tmp_path / f"train0_file_{fold}")
        test0 = read_csv_file(tmp_path / f"test0_file_{fold}")

        train0_inchi, test0_inchi = set(train0.inchikey), set(test0.inchikey)
        assert train0_inchi.isdisjoint(test0_inchi)

        test0_all.append(test0)

    # Check that there's no redundant InChI key in test sets across CV splits
    test0_all = pd.concat(test0_all)
    assert len(test0_all.inchikey) == len(test0_all.inchikey.unique())


def test_02_train_models_RNN(tmp_path):
    train_models_RNN.train_models_RNN(
        representation="SMILES",
        model_type="RNN",
        rnn_type="LSTM",
        embedding_size=32,
        hidden_size=256,
        n_layers=3,
        state_dim=64,
        n_ssm=1,
        n_heads=4,
        exp_factor=4,
        bias=True,
        use_fast_fftconv=False,
        order=2,
        filter_order=64,
        measure="diag-lin",
        mode="diag",
        lr=0.0,
        dropout=0,
        batch_size=64,
        learning_rate=0.001,
        max_epochs=3,
        patience=5000,
        log_every_steps=100,
        log_every_epochs=1,
        sample_mols=100,
        input_file=test_dir
        / "0/prior/inputs/train_LOTUS_truncated_SMILES_0.smi",
        vocab_file=test_dir
        / "0/prior/inputs/train_LOTUS_truncated_SMILES_0.vocabulary",
        model_file=tmp_path / "LOTUS_truncated_SMILES_0_0_model_LSTM.pt",
        loss_file=tmp_path / "LOTUS_truncated_SMILES_0_0_loss.csv",
        smiles_file=None,
    )
    # Model loss values can vary between platforms and architectures,
    # so we simply ensure that this step runs without errors.


def test_02_train_models_conditional_RNN(tmp_path):
    train_models_RNN.train_models_RNN(
        representation="SMILES",
        model_type="RNN",
        rnn_type="LSTM",
        embedding_size=32,
        hidden_size=256,
        n_layers=2,
        state_dim=64,
        n_ssm=1,
        n_heads=4,
        exp_factor=4,
        bias=True,
        use_fast_fftconv=False,
        order=2,
        filter_order=64,
        measure="diag-lin",
        mode="diag",
        lr=0.0,
        dropout=0,
        batch_size=64,
        learning_rate=0.001,
        max_epochs=3,
        patience=5000,
        log_every_steps=100,
        log_every_epochs=1,
        sample_mols=100,
        input_file=test_dir
        / "0/prior/inputs/train_LOTUS_truncated_SMILES_0.smi",
        vocab_file=test_dir
        / "0/prior/inputs/train_LOTUS_truncated_SMILES_0.vocabulary",
        model_file=tmp_path / "LOTUS_truncated_SMILES_0_0_model_cLSTM.pt",
        loss_file=tmp_path / "LOTUS_truncated_SMILES_0_0_loss.csv",
        conditional=True,
        smiles_file=None,
    )
    # Model loss values can vary between platforms and architectures,
    # so we simply ensure that this step runs without errors.


def test_02_train_models_S4(tmp_path):
    train_models_RNN.train_models_RNN(
        representation="SMILES",
        model_type="S4",
        rnn_type="S4",
        embedding_size=32,
        hidden_size=256,
        n_layers=2,
        state_dim=64,
        n_ssm=1,
        n_heads=4,
        exp_factor=4,
        bias=True,
        use_fast_fftconv=False,
        order=2,
        filter_order=64,
        measure="diag-lin",
        mode="diag",
        lr=0.0,
        dropout=0,
        batch_size=64,
        learning_rate=0.001,
        max_epochs=3,
        patience=5000,
        log_every_steps=100,
        log_every_epochs=1,
        sample_mols=100,
        input_file=test_dir
        / "0/prior/inputs/train_LOTUS_truncated_SMILES_0.smi",
        vocab_file=test_dir
        / "0/prior/inputs/train_LOTUS_truncated_SMILES_0.vocabulary",
        model_file=tmp_path / "LOTUS_truncated_SMILES_0_0_model_S4.pt",
        loss_file=tmp_path / "LOTUS_truncated_SMILES_0_0_loss.csv",
        smiles_file=None,
    )
    # Model loss values can vary between platforms and architectures,
    # so we simply ensure that this step runs without errors.


def test_02_train_models_H3(tmp_path):
    train_models_RNN.train_models_RNN(
        representation="SMILES",
        model_type="H3",
        rnn_type="H3",
        embedding_size=32,
        hidden_size=256,
        n_layers=2,
        state_dim=64,
        n_ssm=2,
        n_heads=2,
        exp_factor=4,
        bias=True,
        use_fast_fftconv=False,
        order=2,
        filter_order=64,
        measure="diag-lin",
        mode="diag",
        lr=0.0,
        dropout=0,
        batch_size=64,
        learning_rate=0.001,
        max_epochs=3,
        patience=5000,
        log_every_steps=100,
        log_every_epochs=1,
        sample_mols=100,
        input_file=test_dir
        / "0/prior/inputs/train_LOTUS_truncated_SMILES_0.smi",
        vocab_file=test_dir
        / "0/prior/inputs/train_LOTUS_truncated_SMILES_0.vocabulary",
        model_file=tmp_path / "LOTUS_truncated_SMILES_0_0_model_H3.pt",
        loss_file=tmp_path / "LOTUS_truncated_SMILES_0_0_loss.csv",
        smiles_file=None,
    )
    # Model loss values can vary between platforms and architectures,
    # so we simply ensure that this step runs without errors.


def test_02_train_models_Hyena(tmp_path):
    train_models_RNN.train_models_RNN(
        representation="SMILES",
        model_type="Hyena",
        rnn_type="Hyena",
        embedding_size=32,
        hidden_size=256,
        n_layers=2,
        state_dim=64,
        n_ssm=2,
        n_heads=2,
        exp_factor=4,
        bias=True,
        use_fast_fftconv=False,
        order=2,
        filter_order=64,
        measure="diag-lin",
        mode="diag",
        lr=0.0,
        dropout=0,
        batch_size=64,
        learning_rate=0.001,
        max_epochs=3,
        patience=5000,
        log_every_steps=100,
        log_every_epochs=1,
        sample_mols=100,
        input_file=test_dir
        / "0/prior/inputs/train_LOTUS_truncated_SMILES_0.smi",
        vocab_file=test_dir
        / "0/prior/inputs/train_LOTUS_truncated_SMILES_0.vocabulary",
        model_file=tmp_path / "LOTUS_truncated_SMILES_0_0_model_Hyena.pt",
        loss_file=tmp_path / "LOTUS_truncated_SMILES_0_0_loss.csv",
        smiles_file=None,
    )
    # Model loss values can vary between platforms and architectures,
    # so we simply ensure that this step runs without errors.


def test_02_train_models_Transformer(tmp_path):
    train_models_RNN.train_models_RNN(
        representation="SMILES",
        model_type="Transformer",
        rnn_type="Transformer",
        embedding_size=32,
        hidden_size=256,
        n_layers=3,
        state_dim=64,
        n_ssm=1,
        n_heads=4,
        exp_factor=4,
        bias=True,
        use_fast_fftconv=False,
        order=2,
        filter_order=64,
        measure="diag-lin",
        mode="diag",
        lr=0.0,
        dropout=0,
        batch_size=64,
        learning_rate=0.001,
        max_epochs=3,
        patience=5000,
        log_every_steps=100,
        log_every_epochs=1,
        sample_mols=100,
        input_file=test_dir
        / "0/prior/inputs/train_LOTUS_truncated_SMILES_0.smi",
        vocab_file=test_dir
        / "0/prior/inputs/train_LOTUS_truncated_SMILES_0.vocabulary",
        model_file=tmp_path / "LOTUS_truncated_SMILES_0_0_model_Transformer.pt",
        loss_file=tmp_path / "LOTUS_truncated_SMILES_0_0_loss.csv",
        smiles_file=None,
    )
    # Model loss values can vary between platforms and architectures,
    # so we simply ensure that this step runs without errors.


def test_03_sample_molecules_RNN(tmp_path):
    output_file = (
        tmp_path / "0/prior/samples/LOTUS_truncated_SMILES_0_0_0_samples.csv"
    )
    sample_molecules_RNN.sample_molecules_RNN(
        representation="SMILES",
        model_type="RNN",
        rnn_type="LSTM",
        embedding_size=32,
        hidden_size=256,
        n_layers=3,
        state_dim=64,
        n_ssm=1,
        n_heads=4,
        exp_factor=4,
        bias=True,
        use_fast_fftconv=False,
        order=2,
        filter_order=64,
        measure="diag-lin",
        mode="diag",
        lr=0.0,
        dropout=0,
        batch_size=64,
        sample_mols=100,
        vocab_file=test_dir
        / "0/prior/inputs/train_LOTUS_truncated_SMILES_0.vocabulary",
        model_file=test_dir
        / "0/prior/models/LOTUS_truncated_SMILES_0_0_model.pt",
        output_file=output_file,
    )
    # Samples and their associated loss values can vary between platforms
    # and architectures, so we simply ensure that we have the requisite number
    # of samples
    assert len(read_csv_file(output_file)) == 100


def test_03_sample_molecules_conditional_RNN(tmp_path):
    output_file = (
        tmp_path / "0/prior/samples/LOTUS_truncated_SMILES_0_0_0_samples.csv"
    )
    sample_molecules_RNN.sample_molecules_RNN(
        representation="SMILES",
        model_type="RNN",
        rnn_type="LSTM",
        embedding_size=32,
        hidden_size=256,
        n_layers=3,
        state_dim=64,
        n_ssm=1,
        n_heads=4,
        exp_factor=4,
        bias=True,
        use_fast_fftconv=False,
        order=2,
        filter_order=64,
        measure="diag-lin",
        mode="diag",
        lr=0.0,
        dropout=0,
        batch_size=64,
        sample_mols=100,
        vocab_file=test_dir
        / "0/prior/inputs/train_LOTUS_truncated_SMILES_0.vocabulary",
        model_file=test_dir
        / "0/prior/models/LOTUS_truncated_SMILES_0_0_model_conditional.pt",
        output_file=output_file,
        conditional=True,
        heldout_file=test_dir
        / "0/prior/inputs/test0_LOTUS_truncated_SMILES_1.smi",
    )
    # Samples and their associated loss values can vary between platforms
    # and architectures, so we simply ensure that we have the requisite number
    # of samples
    assert len(read_csv_file(output_file)) == 100


def test_04_tabulate_molecules(tmp_path):
    train_file = test_dir / "0/prior/inputs/train_LOTUS_truncated_SMILES_0.smi"
    output_file = (
        tmp_path
        / "0/prior/samples/LOTUS_truncated_SMILES_0_0_0_samples_masses.csv"
    )
    tabulate_molecules.tabulate_molecules(
        input_file=test_dir
        / "0/prior/samples/LOTUS_truncated_SMILES_0_0_0_samples.csv",
        representation="SMILES",
        train_file=train_file,
        output_file=output_file,
    )
    assert_checksum_equals(
        output_file,
        test_dir
        / "0/prior/samples/LOTUS_truncated_SMILES_0_0_0_samples_masses.csv",
    )
    train = read_csv_file(train_file)
    result = read_csv_file(output_file)

    # Check that same InChI key does not appear in more than one row
    assert len(result.inchikey) == len(result.inchikey.unique())
    # None of the InChI keys in the tabulated molecules should be in the training set
    assert set(train.inchikey).isdisjoint(set(result.inchikey))


def test_05_collect_tabulated_molecules(tmp_path):
    output_file = (
        tmp_path / "0/prior/samples/LOTUS_truncated_SMILES_0_unique_masses.csv"
    )
    collect_tabulated_molecules.collect_tabulated_molecules(
        input_files=[
            test_dir
            / "0/prior/samples/LOTUS_truncated_SMILES_0_0_0_samples_masses.csv",
            test_dir
            / "0/prior/samples/LOTUS_truncated_SMILES_0_1_0_samples_masses.csv",
        ],
        output_file=output_file,
    )
    assert_checksum_equals(
        output_file,
        test_dir / "0/prior/samples/LOTUS_truncated_SMILES_0_unique_masses.csv",
    )

    result = read_csv_file(output_file)

    # Check that same InChI key does not appear in more than one row
    assert len(result.inchikey) == len(result.inchikey.unique())


def test_06_process_tabulated_molecules(tmp_path):
    output_file = (
        tmp_path
        / "0/prior/samples/LOTUS_truncated_SMILES_processed_freq-avg.csv"
    )
    process_tabulated_molecules.process_tabulated_molecules(
        input_file=[
            test_dir
            / "0/prior/samples/LOTUS_truncated_SMILES_0_unique_masses.csv",
            test_dir
            / "0/prior/samples/LOTUS_truncated_SMILES_1_unique_masses.csv",
            test_dir
            / "0/prior/samples/LOTUS_truncated_SMILES_2_unique_masses.csv",
        ],
        cv_files=[
            test_dir / "0/prior/inputs/train_LOTUS_truncated_SMILES_0.smi",
            test_dir / "0/prior/inputs/train_LOTUS_truncated_SMILES_1.smi",
            test_dir / "0/prior/inputs/train_LOTUS_truncated_SMILES_2.smi",
        ],
        output_file=output_file,
        summary_fn="freq-avg",
        min_freq=1,
    )
    assert_checksum_equals(
        output_file,
        test_dir
        / "0/prior/samples/LOTUS_truncated_SMILES_processed_freq-avg.csv",
    )

    result = read_csv_file(output_file)
    # Check that same InChI key does not appear in more than one row
    assert len(result.inchikey) == len(result.inchikey.unique())


def test_07_write_structural_prior_CV(tmp_path):
    temp_dir = tmp_path / "0/prior/structural_prior"
    set_seed(5831)
    write_structural_prior_CV.write_structural_prior_CV(
        ranks_file=temp_dir / "LOTUS_truncated_SMILES_0_CV_ranks_structure.csv",
        tc_file=temp_dir / "LOTUS_truncated_SMILES_0_CV_tc.csv",
        train_file=test_dir
        / "0/prior/inputs/train_LOTUS_truncated_SMILES_0.smi",
        test_file=test_dir
        / "0/prior/inputs/test0_LOTUS_truncated_SMILES_0.smi",
        pubchem_file=pubchem_tsv_file,
        sample_file=test_dir
        / "0/prior/samples/LOTUS_truncated_SMILES_0_unique_masses.csv",
        carbon_file=test_dir
        / "0/prior/inputs/train0_LOTUS_truncated_SMILES_0_carbon.csv",
        err_ppm=10,
        chunk_size=100000,
        top_n=1,
    )
    assert_checksum_equals(
        temp_dir / "LOTUS_truncated_SMILES_0_CV_ranks_structure.csv",
        test_dir
        / "0/prior/structural_prior/LOTUS_truncated_SMILES_0_CV_ranks_structure.csv",
    )
    assert_checksum_equals(
        temp_dir / "LOTUS_truncated_SMILES_0_CV_tc.csv",
        test_dir
        / "0/prior/structural_prior/LOTUS_truncated_SMILES_0_CV_tc.csv",
    )


def test_08_write_formula_prior_CV(tmp_path):
    set_seed(5831)
    write_formula_prior_CV.write_formula_prior_CV(
        ranks_file=tmp_path / "LOTUS_truncated_SMILES_0_CV_ranks_formula.csv",
        train_file=test_dir
        / "0/prior/inputs/train_LOTUS_truncated_SMILES_0.smi",
        test_file=test_dir
        / "0/prior/inputs/test0_LOTUS_truncated_SMILES_0.smi",
        pubchem_file=pubchem_tsv_file,
        sample_file=test_dir
        / "0/prior/samples/LOTUS_truncated_SMILES_0_unique_masses.csv",
        err_ppm=10,
        chunk_size=100000,
    )
    assert_checksum_equals(
        tmp_path / "LOTUS_truncated_SMILES_0_CV_ranks_formula.csv",
        test_dir
        / "0/prior/structural_prior/LOTUS_truncated_SMILES_0_CV_ranks_formula.csv",
    )


def test_08_write_structural_prior_CV(tmp_path):
    temp_dir = tmp_path / "0/prior/structural_prior/add_carbon"
    set_seed(5831)
    write_structural_prior_CV.write_structural_prior_CV(
        ranks_file=temp_dir
        / "LOTUS_truncated_SMILES_min1_all_freq-avg_CV_ranks_structure.csv",
        tc_file=temp_dir / "LOTUS_truncated_SMILES_min1_all_freq-avg_CV_tc.csv",
        train_file=test_dir
        / "0/prior/inputs/train_LOTUS_truncated_SMILES_all.smi",
        test_file=test_dir
        / "0/prior/inputs/test_LOTUS_truncated_SMILES_all.smi",
        pubchem_file=pubchem_tsv_file,
        sample_file=test_dir
        / "0/prior/samples/LOTUS_truncated_SMILES_processed_freq-avg.csv",
        carbon_file=test_dir
        / "0/prior/inputs/train_LOTUS_truncated_SMILES_carbon_all.csv",
        err_ppm=10,
        chunk_size=100000,
        top_n=1,
    )
    assert_checksum_equals(
        temp_dir
        / "LOTUS_truncated_SMILES_min1_all_freq-avg_CV_ranks_structure.csv",
        test_dir
        / "0/prior/structural_prior/LOTUS_truncated_SMILES_min1_all_freq-avg_CV_ranks_structure.csv",
    )
    assert_checksum_equals(
        temp_dir / "LOTUS_truncated_SMILES_min1_all_freq-avg_CV_tc.csv",
        test_dir
        / "0/prior/structural_prior/LOTUS_truncated_SMILES_min1_all_freq-avg_CV_tc.csv",
    )

    plot.plot(
        evaluation_type="structural_prior_min_freq",
        rank_files=[
            temp_dir
            / "LOTUS_truncated_SMILES_min1_all_freq-avg_CV_ranks_structure.csv"
        ],
        tc_files=[
            temp_dir / "LOTUS_truncated_SMILES_min1_all_freq-avg_CV_tc.csv"
        ],
        output_dir=temp_dir,
    )


def test_unique_inchikeys(tmp_path):
    folds = 3
    for fold in range(folds):
        set_seed(5831)
        create_training_sets.create_training_sets(
            input_file=test_dir / "prior/raw/LOTUS_truncated.txt",
            train0_file=tmp_path / "train0_file_{fold}",
            train_file=tmp_path / "train_file_{fold}",
            vocab_file=tmp_path / "vocabulary_file_{fold}",
            test0_file=tmp_path / "test0_file_{fold}",
            enum_factor=3,
            folds=folds,
            which_fold=fold,
            representation="SMILES",
            min_tc=0,
            max_input_smiles=1000,
        )

        train0_inchi = read_csv_file(tmp_path / f"train0_file_{fold}")[
            "inchikey"
        ]
        train_inchi = read_csv_file(tmp_path / f"train_file_{fold}")["inchikey"]

        # Verifying that both augmented and un-augmented training set has same unique inchikeys
        assert set(train_inchi) == set(train0_inchi)
