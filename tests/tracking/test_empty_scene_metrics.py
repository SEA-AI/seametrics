"""Tests for empty-scene tracking evaluation and fair cross-model exclusion.

Pins behavior from docs/plans/tracking-fused-gt-fixes.md before implementation.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import ClassVar
from unittest.mock import patch

import numpy as np
import pytest

from seametrics.tracking import HOTAMetrics, TrackingMetrics, utils
from seametrics.tracking.track import TrackingMetrics as TrackingMetricsClass

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MOT_COLS = 10


def _det(frame: int, obj_id: int, x1: float, y1: float, x2: float, y2: float) -> list:
    return [frame, obj_id, x1, y1, x2, y2]


def _mot_row(
    frame: int,
    obj_id: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    score: float = 1.0,
) -> list:
    return [frame, obj_id, x1, y1, x2, y2, score, -1, -1, -1]


def _scalar_mot(result: dict, metric: str) -> float:
    return next(iter(result[metric].values()))


@contextmanager
def _fo_patch():
    with (
        patch("seametrics.tracking.utils._FIFTYONE_AVAILABLE", True),
        patch("seametrics.tracking.utils.F", lambda field: field, create=True),
    ):
        yield


class _OneSidedEmptyVideoView:
    """Fake FiftyOne view for keyframe-filtered empty GT or empty pred scenarios."""

    media_type = "video"

    def __init__(self, *, empty_side: str) -> None:
        if empty_side not in {"gt", "pred"}:
            raise ValueError(f"empty_side must be 'gt' or 'pred', got {empty_side!r}")
        self.empty_side = empty_side
        self.selected_fields = None

    def has_frame_field(self, field: str) -> bool:
        return field.startswith(("gt.", "pred.", "sequence"))

    def select_fields(self, fields):
        self.selected_fields = fields
        return self

    def first(self):
        return {"metadata": {"frame_width": 100, "frame_height": 50}}

    def distinct(self, field):
        assert field == "sequence"
        return ["seq-empty"]

    def match(self, _condition):
        return self

    def values(self, field: str) -> list:
        box = [[0.1, 0.2, 0.3, 0.4]]
        if self.empty_side == "gt":
            gt_boxes, gt_ids = [[], []], [[], []]
            pred_boxes, pred_ids, pred_scores = [box, box], [[1], [2]], [[0.9], [0.8]]
        else:
            gt_boxes, gt_ids = [box, box], [[1], [2]]
            pred_boxes, pred_ids, pred_scores = [[], []], [[], []], [[], []]

        values = {
            "frames[].gt.detections.bounding_box": gt_boxes,
            "frames[].gt.detections.index": gt_ids,
            "frames[].pred.detections.bounding_box": pred_boxes,
            "frames[].pred.detections.index": pred_ids,
            "frames[].pred.detections.confidence": pred_scores,
            "frames[].pred.keyframe": [True, True],
        }
        return values[field]


class _BrokenPredVideoView:
    """pred_a has keyframes; pred_b does not — only pred_b should fail."""

    media_type = "video"

    def __init__(self) -> None:
        self.selected_fields = None

    def has_frame_field(self, field: str) -> bool:
        return field.startswith(("gt.", "pred_a.", "pred_b.", "sequence"))

    def select_fields(self, fields):
        self.selected_fields = fields
        return self

    def first(self):
        return {"metadata": {"frame_width": 100, "frame_height": 50}}

    def distinct(self, field):
        assert field == "sequence"
        return ["seq-empty"]

    def match(self, _condition):
        return self

    def values(self, field: str) -> list:
        box = [[0.1, 0.2, 0.3, 0.4]]
        shared = {
            "frames[].gt.detections.bounding_box": [[], []],
            "frames[].gt.detections.index": [[], []],
            "frames[].pred_a.detections.bounding_box": [box, box],
            "frames[].pred_a.detections.index": [[1], [2]],
            "frames[].pred_a.detections.confidence": [[0.9], [0.8]],
            "frames[].pred_a.keyframe": [True, True],
            "frames[].pred_b.detections.bounding_box": [box, box],
            "frames[].pred_b.detections.index": [[1], [2]],
            "frames[].pred_b.detections.confidence": [[0.9], [0.8]],
            "frames[].pred_b.keyframe": [False, False],
        }
        return shared[field]


# ---------------------------------------------------------------------------
# prepare_data_for_det_metrics — empty-array shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gt_boxes,gt_ids,pred_boxes,pred_ids,pred_scores,expected_gt,expected_pred",
    [
        pytest.param(
            [],
            [],
            [[[0.1, 0.1, 0.2, 0.2]]],
            [[1]],
            [[0.9]],
            (0, 10),
            (1, 10),
            id="empty-gt",
        ),
        pytest.param(
            [[[0.1, 0.1, 0.2, 0.2]]],
            [[1]],
            [],
            [],
            [],
            (1, 10),
            (0, 10),
            id="empty-pred",
        ),
        pytest.param([], [], [], [], [], (0, 10), (0, 10), id="both-empty"),
    ],
)
def test_prepare_data_when_side_empty_returns_2d_array(
    gt_boxes,
    gt_ids,
    pred_boxes,
    pred_ids,
    pred_scores,
    expected_gt,
    expected_pred,
):
    gt, pred = utils.prepare_data_for_det_metrics(
        gt_bboxes_per_frame=gt_boxes,
        gt_track_ids_per_frame=gt_ids,
        dt_bboxes_per_frame=pred_boxes,
        dt_track_ids_per_frame=pred_ids,
        dt_scores_per_frame=pred_scores,
        img_w=100,
        img_h=100,
    )

    assert gt.shape == expected_gt
    assert pred.shape == expected_pred
    assert gt.ndim == 2
    assert pred.ndim == 2


def test_prepare_data_when_track_id_none_returns_2d_empty_gt():
    gt, pred = utils.prepare_data_for_det_metrics(
        gt_bboxes_per_frame=[[[0.1, 0.1, 0.2, 0.2]]],
        gt_track_ids_per_frame=[[None]],
        dt_bboxes_per_frame=[[[0.1, 0.1, 0.2, 0.2]]],
        dt_track_ids_per_frame=[[10]],
        dt_scores_per_frame=[[0.9]],
        img_w=100,
        img_h=100,
    )

    assert gt.shape == (0, _MOT_COLS)
    assert pred.shape == (1, _MOT_COLS)


def test_tracking_metrics_update_when_prepare_data_empty_pred_accepts_arrays():
    gt, pred = utils.prepare_data_for_det_metrics(
        gt_bboxes_per_frame=[[[0.1, 0.1, 0.2, 0.2]], [[0.2, 0.2, 0.3, 0.4]]],
        gt_track_ids_per_frame=[[1], [1]],
        dt_bboxes_per_frame=[[], []],
        dt_track_ids_per_frame=[[], []],
        dt_scores_per_frame=[[], []],
        img_w=100,
        img_h=100,
    )

    metric = TrackingMetricsClass()
    metric.update(gt, pred, "seq")

    result = metric.compute("seq")
    assert _scalar_mot(result, "num_misses") == 2
    assert _scalar_mot(result, "recall") == pytest.approx(0.0)
    assert math.isnan(_scalar_mot(result, "precision"))


def test_tracking_metrics_update_when_prepare_data_empty_gt_accepts_arrays():
    gt, pred = utils.prepare_data_for_det_metrics(
        gt_bboxes_per_frame=[[], []],
        gt_track_ids_per_frame=[[], []],
        dt_bboxes_per_frame=[[[0.1, 0.1, 0.2, 0.2]], [[0.2, 0.2, 0.3, 0.4]]],
        dt_track_ids_per_frame=[[1], [2]],
        dt_scores_per_frame=[[0.9], [0.8]],
        img_w=100,
        img_h=100,
    )

    metric = TrackingMetricsClass()
    metric.update(gt, pred, "seq")

    result = metric.compute("seq")
    assert _scalar_mot(result, "num_false_positives") == 2
    assert _scalar_mot(result, "precision") == pytest.approx(0.0)
    assert math.isnan(_scalar_mot(result, "recall"))
    assert _scalar_mot(result, "mota") == float("-inf")


class TestTrackingBothEmpty:
    """Both sides empty — motmetrics ratio metrics are undefined."""

    def setup_method(self):
        gt = np.empty((0, 6))
        pred = np.empty((0, 6))
        self.metric = TrackingMetricsClass()
        self.metric.update(gt, pred, "seq")

    def test_num_frames_is_zero(self):
        result = self.metric.compute("seq")
        assert _scalar_mot(result, "num_frames") == 0

    def test_ratio_metrics_are_nan(self):
        result = self.metric.compute("seq")
        for name in ("mota", "motp", "precision", "recall", "idf1"):
            assert math.isnan(_scalar_mot(result, name))


# ---------------------------------------------------------------------------
# compute_all_metrics_by_sequence — empty scenes compute, hard failures exclude
# ---------------------------------------------------------------------------


class _RecordingMetric:
    def __init__(self) -> None:
        self.updates: list = []
        self.failed_sequences: dict = {}

    def update(self, gt, pred, sequence_name: str) -> None:
        self.updates.append((gt, pred, sequence_name))

    def log_failed_sequence(self, sequence_name, _gt, _pred, exc=None) -> None:
        self.failed_sequences[sequence_name] = str(exc) if exc else "logged"


@pytest.mark.parametrize("empty_side", ["gt", "pred"], ids=["empty-gt", "empty-pred"])
def test_compute_all_metrics_when_one_sided_empty_updates_without_failure(empty_side: str):
    view = _OneSidedEmptyVideoView(empty_side=empty_side)

    with _fo_patch():
        instances, excluded = utils.compute_all_metrics_by_sequence(
            view=view,
            gt_field="gt",
            pred_fields=["pred"],
            metrics=[(TrackingMetrics, {}), (HOTAMetrics, {})],
        )

    for metric_name in ("TrackingMetrics", "HOTAMetrics"):
        instance = instances["pred"][metric_name]
        assert instance.failed_sequences == {}
        assert len(instance.updates) == 1
        _gt, _pred, seq = instance.updates[0]
        assert seq == "seq-empty"
        assert _gt.ndim == 2
        assert _pred.ndim == 2

    assert excluded == set()


def test_get_excluded_sequences_when_union_collects_all_failed_without_mutation():
    class _Metric:
        def __init__(self, failures: dict) -> None:
            self.failed_sequences = dict(failures)

    metric_a = _Metric({})
    metric_b = _Metric({"seq-empty": "No keyframe data for: ['pred_b']"})
    instances = {
        "pred_a": {"TrackingMetrics": metric_a},
        "pred_b": {"TrackingMetrics": metric_b},
    }

    excluded = utils.get_excluded_sequences(instances)

    assert excluded == {"seq-empty"}
    assert metric_a.failed_sequences == {}
    assert metric_b.failed_sequences == {"seq-empty": "No keyframe data for: ['pred_b']"}


def test_compute_all_metrics_when_keyframes_missing_excludes_without_updating_any_model():
    view = _BrokenPredVideoView()

    with _fo_patch():
        instances, excluded = utils.compute_all_metrics_by_sequence(
            view=view,
            gt_field="gt",
            pred_fields=["pred_a", "pred_b"],
            metrics=[(_RecordingMetric, {})],
        )

    assert excluded == {"seq-empty"}
    for pred_field in ("pred_a", "pred_b"):
        instance = instances[pred_field]["_RecordingMetric"]
        assert instance.updates == []
        assert "seq-empty" in instance.failed_sequences


def test_comparison_sequence_list_when_excluded_omits_failed_sequences():
    class _Metrics:
        accumulators: ClassVar = {"seq-ok": None, "seq-bad": None}
        RESULT_LAYOUT = "nested"
        failed_sequences: ClassVar = {"seq-bad": "hard failure"}

        def compute(self, sequence=None):
            names = sequence if isinstance(sequence, (list, tuple)) else [sequence]
            return {
                "mota": {name: 0.5 for name in names}
                | {"OVERALL": 0.5},
            }

    metrics = _Metrics()
    excluded = utils.get_excluded_sequences({"model": {"TrackingMetrics": metrics}})
    valid = [s for s in metrics.accumulators if s not in excluded]

    df = utils.results_to_df(metrics, sequence_list=valid)

    assert set(df["sequence"]) == {"seq-ok", utils.OVERALL_LABEL}
    assert "seq-bad" not in df["sequence"].values
