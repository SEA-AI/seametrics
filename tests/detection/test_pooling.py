"""Tests for seametrics.detection.pooling, on hand-written per-sequence results."""

import numpy as np
import pytest

from seametrics.detection.pooling import OVERALL_KEY, aggregate_sequence_results


def _result(tp, fp, fn, fpi=0, n_imgs=1, duplicates=0, area_label="all"):
    """One sequence's compute() output, holding only what pooling reads."""
    return {
        "metrics": {
            area_label: {
                "range": [0, 1e10],
                "iouThr": "0.50",
                "maxDets": 100,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "duplicates": duplicates,
                "precision": None,  # recomputed, never read
                "recall": None,
                "f1": None,
                "support": None,
                "fpi": fpi,
                "nImgs": n_imgs,
            }
        }
    }


def _pooled(*results: dict):
    pooled = aggregate_sequence_results({f"seq{i}": r for i, r in enumerate(results)})
    return pooled["metrics"]["all"]


def test_counts_are_summed_and_ratios_recomputed():
    """(1 tp, 1 fp, 0 fn) + (0, 1, 2): p = r = f1 = 1/3, not the 0.25 mean."""
    overall = _pooled(_result(1, 1, 0, fpi=1, n_imgs=2), _result(0, 1, 2, n_imgs=2))
    assert (overall["tp"], overall["fp"], overall["fn"]) == (1, 2, 2)
    assert (overall["support"], overall["fpi"], overall["nImgs"]) == (3, 1, 4)
    assert overall["precision"] == overall["recall"] == pytest.approx(1 / 3)
    assert overall["f1"] == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        pytest.param((0, 0, 2), (-1, 0.0, -1), id="no-predictions"),
        pytest.param((0, 2, 0), (0.0, -1, -1), id="no-ground-truth"),
        pytest.param((0, 0, 0), (-1, -1, -1), id="empty"),
    ],
)
def test_undefined_ratios_use_the_cocoeval_sentinel(counts, expected):
    overall = _pooled(_result(*counts))
    assert (overall["precision"], overall["recall"], overall["f1"]) == expected


def test_class_agnostic_results_are_python_scalars():
    overall = _pooled(_result(1, 0, 0), _result(0, 0, 1))
    for key in ("tp", "fp", "fn", "duplicates", "support", "fpi", "nImgs"):
        assert type(overall[key]) is int, key
    for key in ("precision", "recall", "f1"):
        assert type(overall[key]) is float, key


def test_per_class_counts_are_summed_position_by_position():
    overall = _pooled(
        _result(np.array([1, 0]), np.array([0, 0]), np.array([0, 0])),
        _result(np.array([0, 1]), np.array([0, 0]), np.array([0, 0])),
    )
    np.testing.assert_array_equal(overall["tp"], [1, 1])
    np.testing.assert_array_equal(overall["precision"], [1.0, 1.0])


def test_existing_overall_is_skipped_so_pooling_is_idempotent():
    results = {"a": _result(1, 0, 0), "b": _result(0, 1, 1)}
    results[OVERALL_KEY] = aggregate_sequence_results(results)
    again = aggregate_sequence_results(results)
    assert again == results[OVERALL_KEY]
    assert set(again) == {"metrics"}


@pytest.mark.parametrize(
    ("results", "match"),
    [
        pytest.param({}, "No sequence results to pool", id="empty"),
        pytest.param(
            {OVERALL_KEY: _result(1, 0, 0)}, "No sequence results", id="only-overall"
        ),
        pytest.param(
            {"a": _result(1, 0, 0), "b": _result(1, 0, 0, area_label="large")},
            "disagree on area-range labels",
            id="area-labels",
        ),
        pytest.param(
            {"a": _result(1, 0, 0), "b": _result(np.array([1, 0]), 0, 0)},
            "Inconsistent `tp` shapes",
            id="class-shapes",
        ),
    ],
)
def test_unpoolable_results_raise(results, match):
    with pytest.raises(ValueError, match=match):
        aggregate_sequence_results(results)
