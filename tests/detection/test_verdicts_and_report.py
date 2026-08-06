"""Tests for per-detection verdicts, tagging, and the report builder.

The verdicts are the per-detection view of the same matching the aggregate
counts summarise, so the property worth pinning is that the two reconcile: the
number of TP verdicts must equal ``tp`` and the number of FP verdicts ``fp``.
"""

from collections import Counter

import fiftyone as fo
import numpy as np
import pytest

from seametrics.detection import PrecisionRecallF1Support
from seametrics.detection.report import DetectionReport, _pool_comparable
from seametrics.detection.utils import (
    OVERALL_KEY,
    payload_to_det_metrics_by_sequence,
    payload_to_detection_verdicts,
    tag_detections,
)
from seametrics.payload import Payload, Resolution, Sequence

BOX = [0.1, 0.1, 0.2, 0.2]
FAR = [0.7, 0.7, 0.1, 0.1]
LOW_IOU = 1e-9


def _gt(box=BOX):
    return fo.Detection(label="BOAT", bounding_box=box)


def _pred(box=BOX, confidence=0.9):
    return fo.Detection(label="BOAT", bounding_box=box, confidence=confidence)


def _payload(gt_frames, pred_frames, keyframes=None, model="model"):
    fields = {"gt": gt_frames, model: pred_frames}
    if keyframes is not None:
        fields["keyframes"] = {model: keyframes}
    return Payload(
        dataset="synthetic",
        models=[model],
        gt_field_name="gt",
        sequences={
            "seq": Sequence(resolution=Resolution(height=100, width=100), **fields)
        },
    )


class TestDetectionVerdicts:
    """PrecisionRecallF1Support.detection_verdicts."""

    def _metric(self, preds, target, **kwargs: object):
        metric = PrecisionRecallF1Support(
            box_format="xywh", iou_thresholds=[LOW_IOU], **kwargs
        )
        metric.update(preds, target)
        return metric, metric.compute()

    def test_hit_and_miss_are_classified(self):
        """One overlapping prediction, one nowhere near the ground truth."""
        preds = [
            dict(
                boxes=np.array([[10.0, 10.0, 20.0, 20.0], [300.0, 300.0, 20.0, 20.0]]),
                scores=np.array([0.9, 0.7]),
                labels=np.array([0, 0]),
            )
        ]
        target = [
            dict(boxes=np.array([[12.0, 12.0, 20.0, 20.0]]), labels=np.array([0]))
        ]
        metric, _ = self._metric(preds, target)

        assert metric.detection_verdicts() == {(0, 0): "TP", (0, 1): "FP"}

    def test_verdicts_reconcile_with_the_counts(self):
        """The whole point: the per-detection view must sum to the aggregate."""
        preds = [
            dict(
                boxes=np.array([[10.0, 10.0, 20.0, 20.0], [300.0, 300.0, 20.0, 20.0]]),
                scores=np.array([0.9, 0.7]),
                labels=np.array([0, 0]),
            ),
            dict(
                boxes=np.array([[50.0, 50.0, 10.0, 10.0]]),
                scores=np.array([0.6]),
                labels=np.array([0]),
            ),
        ]
        target = [
            dict(boxes=np.array([[12.0, 12.0, 20.0, 20.0]]), labels=np.array([0])),
            dict(boxes=np.array([[400.0, 400.0, 10.0, 10.0]]), labels=np.array([0])),
        ]
        metric, results = self._metric(preds, target)
        counts = Counter(metric.detection_verdicts().values())
        summary = results["metrics"]["all"]

        assert counts["TP"] == summary["tp"] == 1
        assert counts["FP"] == summary["fp"] == 2

    def test_indices_address_the_prediction_list(self):
        """Second image, third detection is addressed as (1, 2)."""
        preds = [
            dict(boxes=np.array([]), scores=np.array([]), labels=np.array([])),
            dict(
                boxes=np.array(
                    [
                        [300.0, 300.0, 5.0, 5.0],
                        [310.0, 310.0, 5.0, 5.0],
                        [10.0, 10.0, 20.0, 20.0],
                    ]
                ),
                scores=np.array([0.3, 0.4, 0.9]),
                labels=np.array([0, 0, 0]),
            ),
        ]
        target = [
            dict(boxes=np.array([]), labels=np.array([])),
            dict(boxes=np.array([[12.0, 12.0, 20.0, 20.0]]), labels=np.array([0])),
        ]
        metric, _ = self._metric(preds, target)
        verdicts = metric.detection_verdicts()

        assert verdicts[1, 2] == "TP"  # the overlapping one, listed third
        assert verdicts[1, 0] == "FP"
        assert verdicts[1, 1] == "FP"

    def test_detection_outside_the_area_range_is_ignored(self):
        """Ignored is a third outcome: neither tp nor fp."""
        preds = [
            dict(
                boxes=np.array([[10.0, 10.0, 20.0, 20.0]]),
                scores=np.array([0.9]),
                labels=np.array([0]),
            )
        ]
        target = [
            dict(boxes=np.array([[12.0, 12.0, 20.0, 20.0]]), labels=np.array([0]))
        ]
        metric, _ = self._metric(
            preds, target, area_ranges=[[0, 10]], area_ranges_labels=["tiny"]
        )
        # the 20x20 box is 400 px^2, well outside the 0-10 range
        assert metric.detection_verdicts(area_range_label="tiny") == {(0, 0): "ignored"}

    def test_raises_before_compute(self):
        metric = PrecisionRecallF1Support(box_format="xywh", iou_thresholds=[LOW_IOU])
        with pytest.raises(RuntimeError, match="Call compute"):
            metric.detection_verdicts()

    def test_unknown_area_range_raises(self):
        preds = [
            dict(
                boxes=np.array([[10.0, 10.0, 20.0, 20.0]]),
                scores=np.array([0.9]),
                labels=np.array([0]),
            )
        ]
        target = [
            dict(boxes=np.array([[12.0, 12.0, 20.0, 20.0]]), labels=np.array([0]))
        ]
        metric, _ = self._metric(preds, target)
        with pytest.raises(ValueError, match="was not evaluated"):
            metric.detection_verdicts(area_range_label="nope")

    def test_unknown_iou_threshold_raises(self):
        preds = [
            dict(
                boxes=np.array([[10.0, 10.0, 20.0, 20.0]]),
                scores=np.array([0.9]),
                labels=np.array([0]),
            )
        ]
        target = [
            dict(boxes=np.array([[12.0, 12.0, 20.0, 20.0]]), labels=np.array([0]))
        ]
        metric, _ = self._metric(preds, target)
        with pytest.raises(ValueError, match="was not evaluated"):
            metric.detection_verdicts(iou_threshold=0.5)


class TestPayloadDetectionVerdicts:
    """Mapping verdicts back to fiftyone detection ids."""

    def test_keyed_by_detection_id(self):
        hit, miss = _pred(BOX), _pred(FAR, 0.5)
        payload = _payload([[_gt()]], [[hit, miss]])

        verdicts = payload_to_detection_verdicts(payload, iou_thresholds=[LOW_IOU])

        assert verdicts["seq"] == {hit.id: "TP", miss.id: "FP"}

    def test_reconciles_with_the_sequence_metrics(self):
        gt_frames = [[_gt()], [], [_gt()]]
        pred_frames = [[_pred(BOX), _pred(FAR, 0.4)], [_pred(FAR, 0.3)], []]
        payload = _payload(gt_frames, pred_frames)

        counts = Counter(
            payload_to_detection_verdicts(payload, iou_thresholds=[LOW_IOU])[
                "seq"
            ].values()
        )
        summary = payload_to_det_metrics_by_sequence(
            payload, iou_thresholds=[LOW_IOU], include_overall=False
        )["seq"]["metrics"]["all"]

        assert counts["TP"] == summary["tp"]
        assert counts["FP"] == summary["fp"]

    def test_keyframes_only_leaves_other_frames_unjudged(self):
        """A detection on a non-keyframe was never evaluated, so has no verdict."""
        on_keyframe, off_keyframe = _pred(BOX), _pred(BOX, 0.5)
        payload = _payload(
            [[_gt()], [_gt()]],
            [[on_keyframe], [off_keyframe]],
            keyframes=[True, False],
        )

        verdicts = payload_to_detection_verdicts(
            payload, keyframes_only=True, iou_thresholds=[LOW_IOU]
        )["seq"]

        assert verdicts == {on_keyframe.id: "TP"}
        assert off_keyframe.id not in verdicts

    def test_unmapped_labels_do_not_shift_the_id_alignment(self):
        """A dropped detection must not make later verdicts address the wrong id."""
        dropped = fo.Detection(label="UNKNOWN", bounding_box=FAR, confidence=0.5)
        kept = _pred(BOX)
        payload = _payload([[_gt()]], [[dropped, kept]])

        verdicts = payload_to_detection_verdicts(
            payload,
            label_mapping={"BOAT": 1},
            class_agnostic=False,
            iou_thresholds=[LOW_IOU],
        )["seq"]

        assert verdicts == {kept.id: "TP"}


class TestTagDetections:
    """Writing verdicts onto a dataset as label tags."""

    @pytest.fixture
    def dataset(self):
        name = "_seametrics_tag_test"
        if name in fo.list_datasets():
            fo.delete_dataset(name)
        yield fo.Dataset(name)
        fo.delete_dataset(name)

    def _image_dataset(self, ds, predictions):
        sample = fo.Sample(filepath="/tmp/_seametrics_fake.png")
        sample["model"] = fo.Detections(detections=predictions)
        ds.add_sample(sample)
        return ds

    def test_tags_are_written(self, dataset):
        hit, miss = _pred(BOX), _pred(FAR, 0.5)
        self._image_dataset(dataset, [hit, miss])

        counts = tag_detections(
            dataset, "model", {"seq": {hit.id: "TP", miss.id: "FP"}}
        )

        assert counts["TP"] == 1
        assert counts["FP"] == 1
        stored = {d.id: d.tags for d in dataset.first()["model"].detections}
        assert stored[hit.id] == ["TP"]
        assert stored[miss.id] == ["FP"]

    def test_rerunning_replaces_rather_than_accumulates(self, dataset):
        hit = _pred(BOX)
        self._image_dataset(dataset, [hit])
        verdicts = {"seq": {hit.id: "TP"}}

        tag_detections(dataset, "model", verdicts)
        tag_detections(dataset, "model", verdicts)

        assert dataset.first()["model"].detections[0].tags == ["TP"]

    def test_unrelated_tags_survive(self, dataset):
        hit = _pred(BOX)
        hit.tags = ["reviewed"]
        self._image_dataset(dataset, [hit])

        tag_detections(dataset, "model", {"seq": {hit.id: "TP"}})

        assert dataset.first()["model"].detections[0].tags == ["reviewed", "TP"]

    def test_ignored_is_not_tagged_by_default(self, dataset):
        skipped = _pred(BOX)
        self._image_dataset(dataset, [skipped])

        counts = tag_detections(dataset, "model", {"seq": {skipped.id: "ignored"}})

        assert counts.get("ignored", 0) == 0
        assert dataset.first()["model"].detections[0].tags == []

    def test_ignored_can_be_opted_into(self, dataset):
        skipped = _pred(BOX)
        self._image_dataset(dataset, [skipped])

        tag_detections(
            dataset,
            "model",
            {"seq": {skipped.id: "ignored"}},
            tags=("TP", "FP", "ignored"),
        )

        assert dataset.first()["model"].detections[0].tags == ["ignored"]


class TestPoolComparable:
    """Pooling must cover the same sequences for every model."""

    def _result(self, tp, fp, fn):
        return {
            "metrics": {
                "all": {
                    "range": [0, 1e10],
                    "iouThr": "1e-09",
                    "maxDets": 100,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "duplicates": 0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": -1,
                    "support": tp + fn,
                    "fpi": 0,
                    "nImgs": 1,
                }
            }
        }

    def test_pools_only_the_intersection(self):
        per_sequence = {
            "a": {"s1": self._result(1, 0, 0), "s2": self._result(2, 0, 0)},
            "b": {"s1": self._result(3, 0, 0)},  # never evaluated s2
        }

        overall, pooled = _pool_comparable(per_sequence, ["a", "b"])

        assert pooled == ["s1"]
        assert overall["a"]["metrics"]["all"]["tp"] == 1  # s2 excluded
        assert overall["b"]["metrics"]["all"]["tp"] == 3

    def test_no_shared_sequences_yields_nothing(self):
        per_sequence = {
            "a": {"s1": self._result(1, 0, 0)},
            "b": {"s2": self._result(1, 0, 0)},
        }
        assert _pool_comparable(per_sequence, ["a", "b"]) == ({}, [])

    def test_single_model_pools_everything_it_has(self):
        per_sequence = {"a": {"s1": self._result(1, 0, 0), "s2": self._result(2, 0, 0)}}
        overall, pooled = _pool_comparable(per_sequence, ["a"])
        assert pooled == ["s1", "s2"]
        assert overall["a"]["metrics"]["all"]["tp"] == 3


class TestDetectionReport:
    """The dataclass wrapper."""

    def _report(self):
        results = payload_to_det_metrics_by_sequence(
            _payload([[_gt()]], [[_pred()]]),
            iou_thresholds=[LOW_IOU],
            include_overall=False,
        )
        return DetectionReport(
            models=["model"],
            per_sequence={"model": results},
            overall={"model": results["seq"]},
            pooled_sequences=["seq"],
            skipped={"model": ["dropped"]},
        )

    def test_excluded_sequences_unions_the_skips(self):
        assert self._report().excluded_sequences == ["dropped"]

    def test_to_df_marks_which_rows_were_pooled(self):
        df = self._report().to_df()
        assert list(df["sequence"]) == ["seq", OVERALL_KEY]
        assert list(df["in_overall"]) == [True, False]
        assert set(df["model"]) == {"model"}
