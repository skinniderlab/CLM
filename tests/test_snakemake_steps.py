from pathlib import Path
import pandas as pd
import platform
import socket

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
from clm.functions import assert_checksum_equals, read_csv_file, set_seed

base_dir = Path(__file__).parent.parent

test_dir = base_dir / "tests/test_data/snakemake_output"
dataset = base_dir / "tests/test_data/LOTUS_truncated.txt"
pubchem_tsv_file = base_dir / "tests/test_data/PubChem_truncated.tsv"


def test_00_preprocess(tmp_path):
    preprocess.preprocess(
        input_file=dataset,
        output_file=tmp_path / "preprocessed.smi",
        max_input_smiles=1000,
    )
    assert_checksum_equals(
        tmp_path / "preprocessed.smi", test_dir / "prior/raw/LOTUS_truncated.txt"
    )


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
    assert_checksum_equals(tmp_path / "train0_file_0", tmp_path / "train_file_0")
    assert_checksum_equals(
        tmp_path / "train_file_0",
        test_dir / "0/prior/inputs/train_LOTUS_truncated_SMILES_0.smi",
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
        rnn_type="LSTM",
        embedding_size=32,
        hidden_size=256,
        n_layers=3,
        dropout=0,
        batch_size=64,
        learning_rate=0.001,
        max_epochs=3,
        patience=5000,
        log_every_steps=100,
        log_every_epochs=1,
        sample_mols=100,
        input_file=test_dir / "0/prior/inputs/train_LOTUS_truncated_SMILES_0.smi",
        vocab_file=test_dir
        / "0/prior/inputs/train_LOTUS_truncated_SMILES_0.vocabulary",
        model_file=tmp_path / "LOTUS_truncated_SMILES_0_0_model.pt",
        loss_file=tmp_path / "LOTUS_truncated_SMILES_0_0_loss.csv",
        smiles_file=None,
        minmax_descriptor_file=None,
    )
    # Model loss values can vary between platforms and architectures,
    # so we simply ensure that this step runs without errors.


def test_02_train_models_RNN_conditional(tmp_path):
    loss_file = tmp_path / "LOTUS_truncated_SMILES_0_0_conditional_loss.csv"
    train_models_RNN.train_models_RNN(
        representation="SMILES",
        rnn_type="LSTM",
        embedding_size=32,
        hidden_size=256,
        n_layers=3,
        dropout=0,
        batch_size=64,
        learning_rate=0.001,
        max_epochs=3,
        patience=5000,
        log_every_steps=100,
        log_every_epochs=1,
        sample_mols=100,
        input_file=test_dir / "0/prior/inputs/train_LOTUS_truncated_SMILES_0.smi",
        # input_file=test_dir / "0/prior/inputs/testing_sample.csv", # Testing with csv as input
        vocab_file=test_dir
        / "0/prior/inputs/train_LOTUS_truncated_SMILES_0.vocabulary",
        model_file=tmp_path / "LOTUS_truncated_SMILES_0_0_conditional_model.pt",
        loss_file=loss_file,
        smiles_file=None,
        conditional_rnn=True,
        conditional_emb=False,
        conditional_emb_l=True,
        conditional_dec=False,
        conditional_dec_l=True,
        conditional_h=False,
        minmax_descriptor_file=test_dir / "0/prior/inputs/samples_max_min.csv",
    )

    # TODO: Model losses are platform dependent
    match (platform.system(), socket.gethostname()):
        case ("Darwin", "Adityas-MacBook-Pro.local"):
            assert_checksum_equals(
                loss_file,
                test_dir
                / "0/prior/models/LOTUS_truncated_SMILES_0_0_conditional_loss_darwin.csv",
            )
        case ("Linux", "t15p"):
            assert_checksum_equals(
                loss_file,
                test_dir
                / "0/prior/models/LOTUS_truncated_SMILES_0_0_conditional_loss_linux.csv",
            )


def test_03_sample_molecules_RNN(tmp_path):
    output_file = tmp_path / "0/prior/samples/LOTUS_truncated_SMILES_0_0_0_samples.csv"
    sample_molecules_RNN.sample_molecules_RNN(
        representation="SMILES",
        rnn_type="LSTM",
        embedding_size=32,
        hidden_size=256,
        n_layers=3,
        dropout=0,
        batch_size=64,
        sample_mols=100,
        vocab_file=test_dir
        / "0/prior/inputs/train_LOTUS_truncated_SMILES_0.vocabulary",
        model_file=test_dir / "0/prior/models/LOTUS_truncated_SMILES_0_0_model.pt",
        output_file=output_file,
    )
    # Samples and their associated loss values can vary between platforms
    # and architectures, so we simply ensure that we have the requisite number
    # of samples
    assert len(read_csv_file(output_file)) == 100


def test_03_sample_molecules_RNN_conditional(tmp_path):
    output_file = (
        tmp_path
        / "0/prior/samples/LOTUS_truncated_SMILES_0_0_0_conditional_samples.csv"
    )
    sample_molecules_RNN.sample_molecules_RNN(
        representation="SMILES",
        rnn_type="LSTM",
        embedding_size=32,
        hidden_size=256,
        n_layers=3,
        dropout=0,
        batch_size=64,
        sample_mols=100,
        vocab_file=test_dir
        / "0/prior/inputs/train_LOTUS_truncated_SMILES_0.vocabulary",
        model_file=test_dir
        / "0/prior/models/LOTUS_truncated_SMILES_0_0_conditional_model.pt",
        output_file=output_file,
        conditional_rnn=True,  # Make sure the conditional_rnn,conditional_emb,conditional_emb_l, conditional_dec, conditional_dec_l have the same values present in test_02_train_models_RNN_conditional
        conditional_emb=False,
        conditional_emb_l=True,
        conditional_dec=False,
        conditional_dec_l=True,
        conditional_h=False,
        minmax_descriptor_file=test_dir / "0/prior/inputs/samples_max_min.csv",
        # sample_descriptor_file=test_dir/ "0/prior/inputs/sample_descriptors.csv",#Testing with sample descriptors as input
    )
    assert len(read_csv_file(output_file)) == 100

    match (platform.system(), socket.gethostname()):
        case ("Darwin", "Adityas-MacBook-Pro.local"):
            assert_checksum_equals(
                output_file,
                test_dir
                / "0/prior/samples/LOTUS_truncated_SMILES_0_0_0_conditional_samples_darwin.csv",
            )
        case ("Linux", "t15p"):
            assert_checksum_equals(
                output_file,
                test_dir
                / "0/prior/samples/LOTUS_truncated_SMILES_0_0_0_conditional_samples_linux.csv",
            )


def test_04_tabulate_molecules(tmp_path):
    train_file = test_dir / "0/prior/inputs/train_LOTUS_truncated_SMILES_0.smi"
    output_file = (
        tmp_path / "0/prior/samples/LOTUS_truncated_SMILES_0_0_0_samples_masses.csv"
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
        test_dir / "0/prior/samples/LOTUS_truncated_SMILES_0_0_0_samples_masses.csv",
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
        tmp_path / "0/prior/samples/LOTUS_truncated_SMILES_processed_freq-avg.csv"
    )
    process_tabulated_molecules.process_tabulated_molecules(
        input_file=[
            test_dir / "0/prior/samples/LOTUS_truncated_SMILES_0_unique_masses.csv",
            test_dir / "0/prior/samples/LOTUS_truncated_SMILES_1_unique_masses.csv",
            test_dir / "0/prior/samples/LOTUS_truncated_SMILES_2_unique_masses.csv",
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
        test_dir / "0/prior/samples/LOTUS_truncated_SMILES_processed_freq-avg.csv",
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
        train_file=test_dir / "0/prior/inputs/train_LOTUS_truncated_SMILES_0.smi",
        test_file=test_dir / "0/prior/inputs/test0_LOTUS_truncated_SMILES_0.smi",
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
        test_dir / "0/prior/structural_prior/LOTUS_truncated_SMILES_0_CV_tc.csv",
    )


def test_08_write_formula_prior_CV(tmp_path):
    set_seed(5831)
    write_formula_prior_CV.write_formula_prior_CV(
        ranks_file=tmp_path / "LOTUS_truncated_SMILES_0_CV_ranks_formula.csv",
        train_file=test_dir / "0/prior/inputs/train_LOTUS_truncated_SMILES_0.smi",
        test_file=test_dir / "0/prior/inputs/test0_LOTUS_truncated_SMILES_0.smi",
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
        train_file=test_dir / "0/prior/inputs/train_LOTUS_truncated_SMILES_all.smi",
        test_file=test_dir / "0/prior/inputs/test_LOTUS_truncated_SMILES_all.smi",
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
        temp_dir / "LOTUS_truncated_SMILES_min1_all_freq-avg_CV_ranks_structure.csv",
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
            temp_dir / "LOTUS_truncated_SMILES_min1_all_freq-avg_CV_ranks_structure.csv"
        ],
        tc_files=[temp_dir / "LOTUS_truncated_SMILES_min1_all_freq-avg_CV_tc.csv"],
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

        train0_inchi = read_csv_file(tmp_path / f"train0_file_{fold}")["inchikey"]
        train_inchi = read_csv_file(tmp_path / f"train_file_{fold}")["inchikey"]

        # Verifying that both augmented and un-augmented training set has same unique inchikeys
        assert set(train_inchi) == set(train0_inchi)
