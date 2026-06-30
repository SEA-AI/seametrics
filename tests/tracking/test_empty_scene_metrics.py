"""Tests for empty-scene tracking evaluation (tracking-fused-gt-fixes plan)."""

from __future__ import annotations

import math
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from seametrics.tracking import TrackingMetrics, utils

_MOT_COLS = 10


@contextmanager
def _fo_patch():
    with (
        patch("seametrics.tracking.utils._FIFTYONE_AVAILABLE", True),
        patch("seametrics.tracking.utils.F", lambda field: field, create=True),
    ):
        yield


class _EmptyPredVideoView:
    """GT on two keyframes, tracker output empty — tracker suppressed all."""

    media_type = "video"

    def __init__(self) -> None:
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
        values = {
            "frames[].gt.detections.bounding_box": [box, box],
            "frames[].gt.detections.index": [[1], [2]],
            "frames[].pred.detections.bounding_box": [[], []],
            "frames[].pred.detections.index": [[], []],
            "frames[].pred.detections.confidence": [[], []],
            "frames[].pred.keyframe": [True, True],
        }
        return values[field]


@pytest.mark.parametrize(
    "gt_boxes,gt_ids,pred_boxes,pred_ids,pred_scores,expected_gt,expected_pred",
    [
        pytest.param(
            [],
            [],
            [[[0.1, 0.1, 0.2, 0.2]]],
            [[1]],
            [[0.9]],
            (0, _MOT_COLS),
            (1, _MOT_COLS),
            id="empty-gt",
        ),
        pytest.param(
            [[[0.1, 0.1, 0.2, 0.2]]],
            [[1]],
            [],
            [],
            [],
            (1, _MOT_COLS),
            (0, _MOT_COLS),
            id="empty-pred",
        ),
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


def test_compute_all_metrics_when_empty_pred_scene_computes_without_failure():
    view = _EmptyPredVideoView()

    with _fo_patch():
        instances, excluded = utils.compute_all_metrics_by_sequence(
            view=view,
            gt_field="gt",
            pred_fields=["pred"],
            metrics=[(TrackingMetrics, {"max_iou": 0.5})],
        )

    metric = instances["pred"]["TrackingMetrics"]

    assert excluded == set()
    assert metric.failed_sequences == {}
    assert "seq-empty" in metric.accumulators

    result = metric.compute("seq-empty")
    assert list(result["num_misses"].values())[0] == 2
    assert list(result["recall"].values())[0] == pytest.approx(0.0)
    assert math.isnan(list(result["precision"].values())[0])
