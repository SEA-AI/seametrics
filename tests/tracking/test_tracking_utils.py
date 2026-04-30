import pytest

from seametrics.tracking import utils


class _FakeVideoView:
    media_type = "video"

    def __init__(self):
        self.selected_fields = None

    def has_frame_field(self, field):
        return field == "pred.keyframe" or field.startswith(("gt.", "pred.", "sequence"))

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

    def __init__(self):
        self.video_view = _FakeVideoView()
        self.selected_slice = None

    def select_group_slices(self, group_slice):
        self.selected_slice = group_slice
        return self.video_view

    def match(self, _condition):
        return self


class _RecordingMetric:
    def __init__(self, label=None):
        self.label = label
        self.updates = []
        self.failed_sequences = {}

    def update(self, gt, pred, sequence_name):
        self.updates.append((gt, pred, sequence_name))

    def log_failed_sequence(self, sequence_name, gt, pred, exc=None):
        self.failed_sequences[sequence_name] = str(exc)


class _RejectingMetric(_RecordingMetric):
    def update(self, gt, pred, sequence_name):
        raise ValueError(f"bad sequence {sequence_name}")


class _PairMetric:
    def __init__(self, scale):
        self.scale = scale
        self.updated = None

    def update(self, preds, target):
        self.updated = (preds, target)

    def compute(self):
        preds, target = self.updated
        return {"score": len(preds) + len(target) + self.scale}


def test_tracking_utils_build_sequences_and_format_results():
    view = _FakeGroupView()

    result = utils.compute_metrics(
        view=view,
        gt_field="gt",
        pred_field="pred",
        metric_fn=_PairMetric,
        metric_kwargs={"scale": 1},
    )
    assert result == {"score": 5}

    all_metrics = utils.compute_all_metrics_by_sequence(
        view=view,
        gt_field="gt",
        pred_fields="pred",
        metrics=[
            (_RecordingMetric, {"label": "ok"}),
            (_RejectingMetric, {"label": "rejecting"}),
        ],
    )

    recording = all_metrics["pred"]["_RecordingMetric"]
    rejecting = all_metrics["pred"]["_RejectingMetric"]

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
    assert rejecting.failed_sequences == {"seq-1": "bad sequence seq-1"}

    with pytest.raises(ValueError, match="Duplicate metric class names"):
        utils.compute_all_metrics_by_sequence(
            view=view,
            gt_field="gt",
            pred_fields="pred",
            metrics=[
                (_RecordingMetric, {}),
                (_RecordingMetric, {}),
            ],
            sequence_list=[],
        )

    class _HotaResults:
        accumulators = {"seq-1": None}

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
        accumulators = {"seq-1": None}

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
    assert tracking_df.loc[0, "idf1"] == 0.8
