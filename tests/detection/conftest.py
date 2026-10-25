"""Shared fixtures for detection tests."""

import fiftyone as fo
import pytest


@pytest.fixture
def dataset():
    """Yield a throwaway dataset: non-persistent and uniquely named.

    Yields:
        fo.Dataset: An empty dataset, deleted after the test.
    """
    ds = fo.Dataset()
    yield ds
    ds.delete()
