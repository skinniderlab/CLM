import gzip

import pytest
import seaborn as sns
import pandas as pd
import hashlib
from clm.functions import read_file, read_csv_file, write_to_csv_file, local_seed
from clm.commands.collapse_files import collapse_files


def test_read_file_all(dataset):
    data = read_file(
        dataset, max_lines=None, smile_only=False, stream=False, randomize=False
    )
    assert len(data) == 2501


def test_read_file_5(dataset):
    data = read_file(
        dataset, max_lines=5, smile_only=False, stream=False, randomize=False
    )
    assert len(data) == 5
    assert data[3] == "CS(=O)(=O)C=CCO"  # Line 4


def test_read_file_stream(dataset):
    data = read_file(
        dataset, max_lines=None, smile_only=False, stream=True, randomize=False
    )
    with pytest.raises(TypeError):
        len(data)
    assert len(list(data)) == 2501


def test_read_file_stream_5(dataset):
    data = read_file(
        dataset, max_lines=5, smile_only=False, stream=True, randomize=False
    )
    with pytest.raises(TypeError):
        len(data)
    assert len(list(data)) == 5


def test_read_file_5_randomize(dataset):
    with local_seed(1234):
        data = read_file(
            dataset, max_lines=5, smile_only=False, stream=False, randomize=True
        )
        assert len(data) == 5
        assert data[3] == "CC1(C)C=Cc2c(cc(CO)c(C(=O)O)c2O)O1"  # Line 517


def test_read_file_stream_5_randomize(dataset):
    with local_seed(1234):
        data = read_file(
            dataset, max_lines=5, smile_only=False, stream=True, randomize=True
        )

    with pytest.raises(TypeError):
        len(data)

    data = list(data)
    assert len(data) == 5
    assert data[3] == "CC1(C)C=Cc2c(cc(CO)c(C(=O)O)c2O)O1"  # Line 517


def test_read_file_all_compressed(dataset_compressed):
    data = read_file(
        dataset_compressed,
        max_lines=None,
        smile_only=False,
        stream=False,
        randomize=False,
    )
    assert len(data) == 2501


def test_read_file_5_compressed(dataset_compressed):
    data = read_file(
        dataset_compressed, max_lines=5, smile_only=False, stream=False, randomize=False
    )
    assert len(data) == 5
    assert data[3] == "CS(=O)(=O)C=CCO"  # Line 4


def test_read_file_stream_compressed(dataset_compressed):
    data = read_file(
        dataset_compressed,
        max_lines=None,
        smile_only=False,
        stream=True,
        randomize=False,
    )
    with pytest.raises(TypeError):
        len(data)
    assert len(list(data)) == 2501


def test_read_file_stream_5_compressed(dataset_compressed):
    data = read_file(
        dataset_compressed, max_lines=5, smile_only=True, stream=True, randomize=False
    )
    with pytest.raises(TypeError):
        len(data)
    assert len(list(data)) == 5


def test_read_file_5_randomize_compressed(dataset_compressed):
    with local_seed(1234):
        data = read_file(
            dataset_compressed,
            max_lines=5,
            smile_only=False,
            stream=False,
            randomize=True,
        )
        assert len(data) == 5
        assert data[3] == "CC1(C)C=Cc2c(cc(CO)c(C(=O)O)c2O)O1"  # Line 517


def test_read_file_stream_5_randomize_compressed(dataset_compressed):
    with local_seed(1234):
        data = read_file(
            dataset_compressed,
            max_lines=5,
            smile_only=False,
            stream=True,
            randomize=True,
        )

    with pytest.raises(TypeError):
        len(data)

    data = list(data)
    assert len(data) == 5
    assert data[3] == "CC1(C)C=Cc2c(cc(CO)c(C(=O)O)c2O)O1"  # Line 517


def test_write_csv_uncompressed_dataframe(tmp_path):
    iris = sns.load_dataset("iris")
    write_to_csv_file(tmp_path / "iris.csv", iris)
    df = read_csv_file(tmp_path / "iris.csv")
    pd.testing.assert_frame_equal(iris, df)


def test_write_csv_compressed_dataframe(tmp_path):
    iris = sns.load_dataset("iris")
    write_to_csv_file(tmp_path / "iris.csv.gz", iris)
    df = read_csv_file(tmp_path / "iris.csv.gz")
    pd.testing.assert_frame_equal(iris, df)


def test_write_csv_uncompressed_dataframe_append(tmp_path):
    iris = sns.load_dataset("iris")
    iris2 = iris.copy()
    write_to_csv_file(tmp_path / "iris.csv", iris)
    write_to_csv_file(tmp_path / "iris.csv", iris2, mode="a+")
    df = read_csv_file(tmp_path / "iris.csv")
    pd.testing.assert_frame_equal(pd.concat([iris, iris2]).reset_index(drop=True), df)


def test_write_csv_compressed_dataframe_append(tmp_path):
    iris = sns.load_dataset("iris")
    iris2 = iris.copy()
    write_to_csv_file(tmp_path / "iris.csv.gz", iris)
    write_to_csv_file(tmp_path / "iris.csv.gz", iris2, mode="a+")
    df = read_csv_file(tmp_path / "iris.csv.gz")
    pd.testing.assert_frame_equal(pd.concat([iris, iris2]).reset_index(drop=True), df)


def test_collapse_files1(tmp_path):
    iris = sns.load_dataset("iris")
    write_to_csv_file(tmp_path / "iris1.csv.gz", iris)
    write_to_csv_file(tmp_path / "iris2.csv.gz", iris)

    collapse_files(
        [
            tmp_path / "iris1.csv.gz",
            tmp_path / "iris2.csv.gz",
        ],
        tmp_path / "collapsed.csv",
        has_header=True,
    )

    output_lines = open(tmp_path / "collapsed.csv").readlines()
    assert (
        output_lines[0] == "sepal_length,sepal_width,petal_length,petal_width,species\n"
    )
    assert output_lines[1] == "4.3,3.0,1.1,0.1,setosa\n"

    assert len(output_lines) == 150


def test_collapse_files2(tmp_path):
    iris = sns.load_dataset("iris")
    write_to_csv_file(tmp_path / "iris1.csv.gz", iris)
    write_to_csv_file(tmp_path / "iris2.csv.gz", iris)

    collapse_files(
        [
            tmp_path / "iris1.csv.gz",
            tmp_path / "iris2.csv.gz",
        ],
        tmp_path / "collapsed.csv.gz",
        has_header=True,
    )

    checksum = hashlib.md5(
        "".join(
            [
                line.decode("utf8")
                for line in gzip.open(tmp_path / "collapsed.csv.gz", "r").readlines()
            ]
        ).encode("utf8")
    ).hexdigest()
    assert checksum == "d0d4de624374a8d48d52b8bd3b68f099"


def test_collapse_files3(tmp_path):
    iris = sns.load_dataset("iris")
    write_to_csv_file(tmp_path / "iris1.csv", iris)
    write_to_csv_file(tmp_path / "iris2.csv", iris)

    collapse_files(
        [
            tmp_path / "iris1.csv",
            tmp_path / "iris2.csv",
        ],
        tmp_path / "collapsed.csv",
        has_header=True,
    )

    output_lines = open(tmp_path / "collapsed.csv").readlines()
    assert (
        output_lines[0] == "sepal_length,sepal_width,petal_length,petal_width,species\n"
    )
    assert output_lines[1] == "4.3,3.0,1.1,0.1,setosa\n"

    assert len(output_lines) == 150


def test_collapse_files4(tmp_path):
    iris = sns.load_dataset("iris")
    write_to_csv_file(tmp_path / "iris1.csv", iris)
    write_to_csv_file(tmp_path / "iris2.csv", iris)

    collapse_files(
        [
            tmp_path / "iris1.csv",
            tmp_path / "iris2.csv",
        ],
        tmp_path / "collapsed.csv.gz",
        has_header=True,
    )

    checksum = hashlib.md5(
        "".join(
            [
                line.decode("utf8")
                for line in gzip.open(tmp_path / "collapsed.csv.gz", "r").readlines()
            ]
        ).encode("utf8")
    ).hexdigest()
    assert checksum == "d0d4de624374a8d48d52b8bd3b68f099"
