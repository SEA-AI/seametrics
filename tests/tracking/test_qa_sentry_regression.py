"""Regression tests for QA_SENTRY_2026_03_VIDEO_BB thermal_wide failures.

Fixtures were exported from ``QA_SENTRY_2026_03_VIDEO_BB`` (thermal_wide,
``ground_truth_det_fused_id``) for the four sequences that incorrectly landed
in ``failed_sequences`` on ``develop`` when evaluating:

- ``tracker_baseline_develop_e8f615a_oversea_det``
- ``tracker_fix_prob_e8f615a_oversea_det``
- ``SEA_1052_TPE_tight_172ef21_oversea_det``
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from seametrics.tracking import HOTAMetrics, TrackingMetrics

_FIXTURE_DIR = Path(__file__).parent / "fixtures"
_DATA = np.load(_FIXTURE_DIR / "qa_sentry_2026_03_regression.npz")
_EXPECTED = np.load(_FIXTURE_DIR / "qa_sentry_2026_03_expected.npz")

_CASES = (
    ("seq_0928_164219", "baseline"),
    ("seq_0928_164219", "fix_prob"),
    ("seq_0928_164219", "tpe_tight"),
    ("seq_0929_142127", "baseline"),
    ("seq_0929_142127", "fix_prob"),
    ("seq_0929_142127", "tpe_tight"),
    ("seq_bsh_121534", "baseline"),
    ("seq_bsh_121534", "fix_prob"),
    ("seq_bsh_121534", "tpe_tight"),
    ("seq_proact_163418", "baseline"),
    ("seq_proact_163418", "fix_prob"),
    ("seq_proact_163418", "tpe_tight"),
)


def _load_pair(seq_key: str, model_key: str) -> tuple[np.ndarray, np.ndarray]:
    prefix = f"{seq_key}__{model_key}"
    return _DATA[f"{prefix}__gt"], _DATA[f"{prefix}__pred"]


@pytest.mark.parametrize(("seq_key", "model_key"), _CASES)
def test_qa_sentry_sequences_compute_without_failed_sequences(
    seq_key: str, model_key: str
):
    """Empty-scene QA sequences must update metrics, not log failed_sequences."""
    gt, pred = _load_pair(seq_key, model_key)
    assert gt.ndim == 2
    assert pred.ndim == 2

    mot = TrackingMetrics(max_iou=0.5)
    mot.update(gt, pred, seq_key)
    assert mot.failed_sequences == {}
    assert seq_key in mot.accumulators

    hota = HOTAMetrics()
    hota.update(gt, pred, seq_key)
    assert hota.failed_sequences == {}
    assert seq_key in hota.accumulators


@pytest.mark.parametrize(("seq_key", "model_key"), _CASES)
def test_qa_sentry_sequences_match_exported_metric_counts(
    seq_key: str, model_key: str
):
    """Pin MOT/HOTA outputs from the real QA_SENTRY export run."""
    gt, pred = _load_pair(seq_key, model_key)
    prefix = f"{seq_key}__{model_key}"

    mot = TrackingMetrics(max_iou=0.5)
    mot.update(gt, pred, seq_key)
    mot_res = mot.compute(seq_key)
    assert next(iter(mot_res["num_misses"].values())) == int(
        _EXPECTED[f"{prefix}__mot_num_misses"]
    )
    assert next(iter(mot_res["num_false_positives"].values())) == int(
        _EXPECTED[f"{prefix}__mot_num_fp"]
    )

    hota = HOTAMetrics()
    hota.update(gt, pred, seq_key)
    hota_val = hota.compute(seq_key)["hota"]
    expected_hota = float(_EXPECTED[f"{prefix}__hota"])
    if math.isnan(expected_hota):
        assert math.isnan(hota_val)
    else:
        assert hota_val == pytest.approx(expected_hota)
