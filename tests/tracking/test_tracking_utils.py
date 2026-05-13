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

    def update(self, gt, pred, sequence_name):
        self.updates.append((gt, pred, sequence_name))

    def log_failed_sequence(self, sequence_name, gt, pred, exc=None):
        raise AssertionError(f"unexpected failure for {sequence_name}: {exc}")


def test_compute_all_metrics_by_sequence_uses_group_slice_and_keyframes():
    view = _FakeGroupView()

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
        def __init__(self): pass
        def update(self, gt, pred, seq): raise AssertionError("should not be called")
        def log_failed_sequence(self, seq, gt, pred, exc=None):
            failures.append((seq, exc))

    class _ErrorView(_FakeVideoView):
        def values(self, field):
            if "keyframe" in field:
                raise RuntimeError("field not found")
            return super().values(field)

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
        def __init__(self): pass
        def update(self, gt, pred, seq): raise AssertionError("should not be called")
        def log_failed_sequence(self, seq, gt, pred, exc=None):
            failures.append((seq, exc))

    class _NoKeyframeView(_FakeVideoView):
        def values(self, field):
            if "keyframe" in field:
                return [False, False, False]
            return super().values(field)

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
        def __init__(self): pass
        def update(self, gt, pred, seq): raise AssertionError("should not be called")
        def log_failed_sequence(self, seq, gt, pred, exc=None):
            failures.append(seq)

    class _PartialKeyframeView(_FakeVideoView):
        def values(self, field):
            if "pred_a.keyframe" in field:
                return [True, False, True]
            if "pred_b.keyframe" in field:
                return [False, False, False]
            return super().values(field)

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
