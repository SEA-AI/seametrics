"""Shared fixtures for tracking tests."""

import pytest


@pytest.fixture(autouse=True)
def patch_fiftyone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mark fiftyone as available and stub ViewField for unit tests."""
    monkeypatch.setattr("seametrics.tracking.utils._FIFTYONE_AVAILABLE", True)
    monkeypatch.setattr(
        "seametrics.tracking.utils.F",
        lambda field: field,
        raising=False,
    )
