"""Shared fixtures: synthetic data with the real schema, round-tripped through CSV."""
import pytest

from src import data, split
from tests.make_fake_data import make


@pytest.fixture(scope="session")
def raw_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("raw") / "diabetic_data.csv"
    make(8000, seed=1).to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def df(raw_csv):
    return data.prepare(data.load_raw(raw_csv))


@pytest.fixture(scope="session")
def split_frames(df, tmp_path_factory):
    f = tmp_path_factory.mktemp("split") / "test_patients.csv"
    return split.get_split(df, f)
