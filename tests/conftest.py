from pathlib import Path
import gzip
import pytest
from clm.functions import set_seed

base_dir = Path(__file__).parent.parent
dataset_path = base_dir / "tests/test_data/LOTUS_truncated.txt"


@pytest.fixture(scope="session")
def dataset():
    return dataset_path


@pytest.fixture(scope="session")
def dataset_compressed():
    path = dataset_path.with_suffix(".txt.gz")
    with gzip.open(path, "wb") as g:
        with open(dataset_path, "rb") as f:
            g.write(f.read())
    yield path


@pytest.fixture(scope="function", autouse=True)
def seed():
    """
    Set the random seed to 0 for all tests.
    This prevents individual tests that use set_seed from affecting each other.
    """
    set_seed(0)
