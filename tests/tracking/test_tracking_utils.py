"""Tests for seametrics.tracking.utils."""

from contextlib import contextmanager
from typing import ClassVar
from unittest.mock import patch

import pytest

from seametrics.tracking import utils


class _FakeVideoView:
    media_type = "video"

    def __init__(self) -> None:
        self.selected_fields = None

    def has_frame_field(self, field):
        return field == "pred.keyframe" or field.startswith(
            ("gt.", "pred.", "sequence")
        )

    def select_fields(self, fields):
        self.selected_fields = fields
        return self

    def first(self):
        return {"metadata": {"frame_width": 100, "frame_height": 50}}

    def distinct(self, field):
        assert field == "sequence"
        return ["seq-1"]

    def match(self, _condition):
        return self

    def values(self, field):
        values = {
            "frames[].gt.detections.bounding_box": [
                [[0.1, 0.2, 0.3, 0.4]],
                [[0.2, 0.2, 0.3, 0.4]],
                [[0.4, 0.2, 0.3, 0.4]],
            ],
            "frames[].gt.detections.index": [[1], [2], [3]],
            "frames[].pred.detections.bounding_box": [
                [[0.1, 0.2, 0.3, 0.4]],
                [[0.2, 0.2, 0.3, 0.4]],
                [[0.4, 0.2, 0.3, 0.4]],
            ],
            "frames[].pred.detections.confidence": [[0.9], [0.8], [0.7]],
            "frames[].pred.detections.index": [[10], [20], [30]],
            "frames[].pred.keyframe": [True, False, True],
        }
        return values[field]


class _FakeGroupView:
    media_type = "group"
    default_group_slice = "rgb"
    dataset_name = "dataset"

    def __init__(self) -> None:
        self.video_view = _FakeVideoView()
        self.selected_slice = None

    def select_group_slices(self, group_slice):
        self.selected_slice = group_slice
        return self.video_view

    def match(self, _condition):
        return self


class _RecordingMetric:
    def __init__(self, label=None) -> None:
        self.label = label
        self.updates = []
        self.failed_sequences: dict = {}

    def update(self, gt, pred, sequence_name):
        self.updates.append((gt, pred, sequence_name))

    def log_failed_sequence(self, sequence_name, _gt, _pred, exc=None):
        raise AssertionError(f"unexpected failure for {sequence_name}: {exc}")


@contextmanager
def _fo_patch():
    """Context manager that fakes fiftyone availability without an install."""
    with (
        patch("seametrics.tracking.utils._FIFTYONE_AVAILABLE", True),
        patch("seametrics.tracking.utils.F", lambda field: field, create=True),
    ):
        yield


def test_compute_all_metrics_by_sequence_uses_group_slice_and_keyframes():
    view = _FakeGroupView()

    with _fo_patch():
        all_metrics = utils.compute_all_metrics_by_sequence(
            view=view,
            gt_field="gt",
            pred_fields="pred",
            metrics=[(_RecordingMetric, {"label": "ok"})],
        )

    recording = all_metrics["pred"]["_RecordingMetric"]

    assert view.selected_slice == "rgb"
    assert recording.label == "ok"
    assert len(recording.updates) == 1

    gt, pred, sequence_name = recording.updates[0]
    assert sequence_name == "seq-1"
    assert gt.shape == (2, 10)
    assert pred.shape == (2, 10)
    assert gt[:, 1].tolist() == [1, 3]
    assert pred[:, 1].tolist() == [10, 30]
    assert pred[:, 6].tolist() == [0.9, 0.7]


def test_sequence_skipped_when_keyframe_lookup_raises():
    """_has_keyframes except branch: returns False when values() raises."""
    failures = []

    class _CapturingMetric:
        def __init__(self) -> None:
            self.failed_sequences: dict = {}

        def update(self, _gt, _pred, _seq):
            raise AssertionError("should not be called")

        def log_failed_sequence(self, seq, _gt, _pred, **_kwargs: object):
            failures.append((seq, _kwargs.get("exc")))
            self.failed_sequences[seq] = str(_kwargs.get("exc"))

    class _ErrorView(_FakeVideoView):
        def values(self, field):
            if "keyframe" in field:
                raise RuntimeError("field not found")
            return super().values(field)

    with _fo_patch():
        utils.compute_all_metrics_by_sequence(
            view=_ErrorView(),
            gt_field="gt",
            pred_fields=["pred"],
            metrics=[(_CapturingMetric, {})],
        )

    assert len(failures) == 1
    seq, exc = failures[0]
    assert seq == "seq-1"
    assert "pred" in str(exc)


def test_sequence_skipped_when_no_true_keyframes():
    """_has_keyframes returns False when all keyframe values are False."""
    failures = []

    class _CapturingMetric:
        def __init__(self) -> None:
            self.failed_sequences: dict = {}

        def update(self, _gt, _pred, _seq):
            raise AssertionError("should not be called")

        def log_failed_sequence(self, seq, _gt, _pred, **_kwargs: object):
            failures.append((seq, _kwargs.get("exc")))
            self.failed_sequences[seq] = str(_kwargs.get("exc"))

    class _NoKeyframeView(_FakeVideoView):
        def values(self, field):
            if "keyframe" in field:
                return [False, False, False]
            return super().values(field)

    with _fo_patch():
        utils.compute_all_metrics_by_sequence(
            view=_NoKeyframeView(),
            gt_field="gt",
            pred_fields=["pred"],
            metrics=[(_CapturingMetric, {})],
        )

    assert len(failures) == 1
    assert failures[0][0] == "seq-1"
    assert "pred" in str(failures[0][1])


def test_sequence_skipped_logs_all_pred_fields_when_one_missing():
    """When one pred_field lacks keyframes, all pred_fields are logged as failed."""
    failures = []

    class _CapturingMetric:
        def __init__(self) -> None:
            self.failed_sequences: dict = {}

        def update(self, _gt, _pred, _seq):
            raise AssertionError("should not be called")

        def log_failed_sequence(self, seq, _gt, _pred, **_kwargs: object):
            failures.append(seq)
            self.failed_sequences[seq] = str(_kwargs.get("exc"))

    class _PartialKeyframeView(_FakeVideoView):
        def values(self, field):
            if "pred_a.keyframe" in field:
                return [True, False, True]
            if "pred_b.keyframe" in field:
                return [False, False, False]
            return super().values(field)

    with _fo_patch():
        utils.compute_all_metrics_by_sequence(
            view=_PartialKeyframeView(),
            gt_field="gt",
            pred_fields=["pred_a", "pred_b"],
            metrics=[(_CapturingMetric, {})],
        )

    assert len(failures) == 2
    assert all(seq == "seq-1" for seq in failures)


def test_results_to_df_formats_hota_and_tracking_outputs():
    class _HotaResults:
        accumulators: ClassVar = {"seq-1": None}

        def compute(self, sequence):
            assert sequence == "seq-1"
            return {
                "hota": 0.5,
                "deta": 0.75,
                "assa": 0.25,
                "loca": 1.0,
                "num_unique_objects": 2,
            }

    hota_df = utils.hota_results_to_df(_HotaResults())
    assert hota_df.loc[0, "hota"] == 50
    assert hota_df.loc[0, "deta"] == 75
    assert hota_df.loc[0, "num_unique_objects"] == 2
    assert hota_df.loc[0, "sequence"] == "seq-1"

    class _TrackingResults:
        accumulators: ClassVar = {"seq-1": None}

        def compute(self, sequence):
            assert sequence == "seq-1"
            return {
                "mota": {"seq-1": 0.25},
                "motp": {"seq-1": 0.2},
                "idf1": {"seq-1": 0.8},
            }

    tracking_df = utils.results_to_df(_TrackingResults())
    assert tracking_df.loc[0, "mota"] == 25
    assert tracking_df.loc[0, "motp"] == 80
    assert tracking_df.loc[0, "idf1"] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# prepare_data_for_det_metrics
# ---------------------------------------------------------------------------


def test_prepare_data_basic_single_frame():
    """One GT and one pred in one frame produce (1, 10) arrays each."""
    gt, pred = utils.prepare_data_for_det_metrics(
        gt_bboxes_per_frame=[[[0.1, 0.1, 0.2, 0.2]]],
        gt_track_ids_per_frame=[[1]],
        dt_bboxes_per_frame=[[[0.1, 0.1, 0.2, 0.2]]],
        dt_track_ids_per_frame=[[10]],
        dt_scores_per_frame=[[0.9]],
        img_w=100,
        img_h=100,
    )
    assert gt.shape == (1, 10)
    assert pred.shape == (1, 10)
    assert gt[0, 0] == 1
    assert gt[0, 1] == 1
    assert pred[0, 1] == 10
    assert pred[0, 6] == pytest.approx(0.9)


def test_prepare_data_multi_frame():
    """Multiple frames concatenate correctly."""
    gt, pred = utils.prepare_data_for_det_metrics(
        gt_bboxes_per_frame=[[[0.0, 0.0, 0.5, 0.5]], [[0.5, 0.5, 0.5, 0.5]]],
        gt_track_ids_per_frame=[[1], [2]],
        dt_bboxes_per_frame=[[[0.0, 0.0, 0.5, 0.5]], [[0.5, 0.5, 0.5, 0.5]]],
        dt_track_ids_per_frame=[[10], [20]],
        dt_scores_per_frame=[[0.8], [0.7]],
        img_w=100,
        img_h=100,
    )
    assert gt.shape == (2, 10)
    assert pred.shape == (2, 10)
    assert gt[:, 0].tolist() == [1, 2]
    assert gt[:, 1].tolist() == [1, 2]


def test_prepare_data_none_bbox_skipped():
    """A frame whose bbox list is None must not produce a GT row."""
    gt, _pred = utils.prepare_data_for_det_metrics(
        gt_bboxes_per_frame=[None, [[0.1, 0.1, 0.2, 0.2]]],
        gt_track_ids_per_frame=[None, [1]],
        dt_bboxes_per_frame=[None, [[0.1, 0.1, 0.2, 0.2]]],
        dt_track_ids_per_frame=[None, [10]],
        dt_scores_per_frame=[None, [0.9]],
        img_w=100,
        img_h=100,
    )
    assert gt.shape == (1, 10)
    assert gt[0, 0] == 2


def test_prepare_data_none_track_id_skipped():
    """A detection with track_id=None must not produce a GT row."""
    gt, pred = utils.prepare_data_for_det_metrics(
        gt_bboxes_per_frame=[[[0.1, 0.1, 0.2, 0.2]]],
        gt_track_ids_per_frame=[[None]],
        dt_bboxes_per_frame=[[[0.1, 0.1, 0.2, 0.2]]],
        dt_track_ids_per_frame=[[10]],
        dt_scores_per_frame=[[0.9]],
        img_w=100,
        img_h=100,
    )
    assert gt.shape == (0,)
    assert pred.shape == (1, 10)


def test_prepare_data_already_pixel_boxes_not_rescaled():
    """Boxes with values > 1 are treated as pixel coords and not re-scaled."""
    gt, pred = utils.prepare_data_for_det_metrics(
        gt_bboxes_per_frame=[[[10.0, 10.0, 50.0, 50.0]]],
        gt_track_ids_per_frame=[[1]],
        dt_bboxes_per_frame=[[[10.0, 10.0, 50.0, 50.0]]],
        dt_track_ids_per_frame=[[5]],
        dt_scores_per_frame=[[1.0]],
        img_w=640,
        img_h=480,
    )
    assert gt.shape == (1, 10)
    assert pred.shape == (1, 10)


# ---------------------------------------------------------------------------
# sequence_results_to_df
# ---------------------------------------------------------------------------


def test_sequence_results_to_df_basic():
    """Single sequence, single area range populates one row."""
    results = {
        "seq-1": {
            "metrics": {
                "all": {
                    "range": [0, 1e5],
                    "iouThr": 0.5,
                    "maxDets": 100,
                    "tp": 10,
                    "fp": 2,
                    "fn": 1,
                    "duplicates": 0,
                    "precision": 0.83,
                    "recall": 0.91,
                    "f1": 0.87,
                    "support": 11,
                    "fpi": 0.1,
                    "nImgs": 50,
                }
            }
        }
    }
    df = utils.sequence_results_to_df(results)
    assert len(df) == 1
    assert df.loc[0, "sequence"] == "seq-1"
    assert df.loc[0, "tp"] == 10
    assert df.loc[0, "precision"] == pytest.approx(0.83)


def test_sequence_results_to_df_multiple_sequences_and_ranges():
    """Two sequences x two area ranges produce four rows."""
    area = {
        "range": [0, 1e5],
        "iouThr": 0.5,
        "maxDets": 100,
        "tp": 1,
        "fp": 0,
        "fn": 0,
        "duplicates": 0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "support": 1,
        "fpi": 0.0,
        "nImgs": 10,
    }
    results = {
        f"seq-{i}": {"metrics": {lbl: dict(area) for lbl in ("small", "large")}}
        for i in range(2)
    }
    df = utils.sequence_results_to_df(results)
    assert len(df) == 4


# ---------------------------------------------------------------------------
# classify_num_objects
# ---------------------------------------------------------------------------


def test_classify_num_objects_zero():
    assert utils.classify_num_objects(0) == "zero"


def test_classify_num_objects_one():
    assert utils.classify_num_objects(1) == "one"


def test_classify_num_objects_two():
    assert utils.classify_num_objects(2) == "two"


def test_classify_num_objects_few():
    assert utils.classify_num_objects(4) == "few"


def test_classify_num_objects_many():
    assert utils.classify_num_objects(10) == "many"


def test_classify_num_objects_out_of_range_returns_none():
    assert utils.classify_num_objects(100) is None
