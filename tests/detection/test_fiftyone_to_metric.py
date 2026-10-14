"""End-to-end tests: fiftyone detections -> PrecisionRecallF1Support.

These pin the *downstream consequences* of the fiftyone box/width/height parsing.
Every expected value is hand-computable from the boxes in the test body, so a
reviewer can verify the math by eye.
"""

import fiftyone as fo
import numpy as np
import pytest

from seametrics.detection import PrecisionRecallF1Support
from seametrics.detection.utils import (
    OVERALL_KEY,
    aggregate_sequence_results,
    payload_sequence_to_det_metrics,
    payload_to_det_metric,
    payload_to_det_metrics_by_sequence,
    sequence_results_to_df,
)
from seametrics.payload import Payload, Resolution, Sequence

# A ground-truth / prediction pair with IoU exactly 1/3 on a 100x100 image.
#
#   gt   rel [0.1, 0.1, 0.2, 0.2] -> px xywh [10, 10, 20, 20] -> xyxy [10, 10, 30, 30]
#   pred rel [0.2, 0.1, 0.2, 0.2] -> px xywh [20, 10, 20, 20] -> xyxy [20, 10, 40, 30]
#
#   intersection = 10 * 20 = 200, union = 400 + 400 - 200 = 600, IoU = 1/3
GT_BOX = [0.1, 0.1, 0.2, 0.2]
PRED_BOX = [0.2, 0.1, 0.2, 0.2]

# Area ranges split at 300 px², so a 20x20 px (400 px²) box belongs to "large".
AREA_RANGES = [[0, 300], [300, 1e10]]
AREA_LABELS = ["small", "large"]

# Same construction on a 100x400 image; see TestNonSquareResolution.
NONSQUARE_GT = [0.0, 0.0, 0.5, 0.5]
NONSQUARE_PRED = [0.0, 0.25, 0.5, 0.5]


def _convert(gt_frames, pred_frames, width, height):
    """Run both sides of a sequence through the fiftyone converter.

    Args:
        gt_frames: Per-frame lists of ground-truth ``fo.Detection`` objects.
        pred_frames: Per-frame lists of prediction ``fo.Detection`` objects.
        width: Sequence width in pixels.
        height: Sequence height in pixels.

    Returns:
        ``(predictions, references)`` in ``PrecisionRecallF1Support`` input format.
    """
    preds = payload_sequence_to_det_metrics(pred_frames, width, height)
    refs = payload_sequence_to_det_metrics(gt_frames, width, height, is_gt=True)
    return preds, refs


def _gt(box, area=None):
    """Build a ground-truth detection, optionally with an explicit area."""
    kwargs = {} if area is None else {"area": area}
    return fo.Detection(label="BOAT", bounding_box=box, **kwargs)


def _pred(box, confidence=0.9):
    """Build a prediction detection."""
    return fo.Detection(label="BOAT", bounding_box=box, confidence=confidence)


def _compute(preds, refs, **metric_kwargs: object):
    """Feed converted data to the metric using the xywh contract."""
    metric_kwargs.setdefault("box_format", "xywh")
    metric = PrecisionRecallF1Support(**metric_kwargs)
    metric.update(preds, refs)
    return metric.compute()


class TestIouFromConvertedBoxes:
    """The converted boxes must reproduce the hand-computed IoU of 1/3."""

    def test_match_when_threshold_below_iou(self):
        preds, refs = _convert(
            [[_gt(GT_BOX, area=400.0)]], [[_pred(PRED_BOX)]], 100, 100
        )
        res = _compute(preds, refs, iou_thresholds=[0.3])["metrics"]["all"]
        assert (res["tp"], res["fp"], res["fn"]) == (1, 0, 0)
        assert res["precision"] == pytest.approx(1.0)
        assert res["recall"] == pytest.approx(1.0)
        assert res["f1"] == pytest.approx(1.0)
        assert res["support"] == 1

    def test_no_match_when_threshold_above_iou(self):
        preds, refs = _convert(
            [[_gt(GT_BOX, area=400.0)]], [[_pred(PRED_BOX)]], 100, 100
        )
        res = _compute(preds, refs, iou_thresholds=[0.35])["metrics"]["all"]
        assert (res["tp"], res["fp"], res["fn"]) == (0, 1, 1)
        assert res["precision"] == pytest.approx(0.0)
        assert res["recall"] == pytest.approx(0.0)
        assert res["f1"] == -1  # undefined: precision + recall == 0
        assert res["support"] == 1

    def test_identical_boxes_match_at_every_threshold(self):
        preds, refs = _convert([[_gt(GT_BOX, area=400.0)]], [[_pred(GT_BOX)]], 100, 100)
        res = _compute(preds, refs, iou_thresholds=[0.95])["metrics"]["all"]
        assert (res["tp"], res["fp"], res["fn"]) == (1, 0, 0)

    def test_xywh_is_the_required_box_format(self):
        """Reading the converter output as xyxy destroys the match.

        Under xyxy, the prediction [20, 10, 20, 20] becomes a zero-width box, so
        the IoU collapses to 0 and the true positive is lost. This is why callers
        must pass ``box_format="xywh"``.
        """
        preds, refs = _convert(
            [[_gt(GT_BOX, area=400.0)]], [[_pred(PRED_BOX)]], 100, 100
        )
        as_xywh = _compute(preds, refs, iou_thresholds=[0.3])["metrics"]["all"]
        as_xyxy = _compute(preds, refs, iou_thresholds=[0.3], box_format="xyxy")[
            "metrics"
        ]["all"]
        assert as_xywh["tp"] == 1
        assert as_xyxy["tp"] == 0


class TestNonSquareResolution:
    """Height must scale y-coordinates; width must scale x-coordinates.

    On a 100x400 image:
      gt   rel [0.0, 0.00, 0.5, 0.5] -> px xywh [0,   0, 50, 200]
      pred rel [0.0, 0.25, 0.5, 0.5] -> px xywh [0, 100, 50, 200]
      intersection = 50 * 100 = 5000, union = 20000 - 5000 = 15000, IoU = 1/3
    """

    def test_match_when_threshold_below_iou(self):
        preds, refs = _convert(
            [[_gt(NONSQUARE_GT)]], [[_pred(NONSQUARE_PRED)]], 100, 400
        )
        res = _compute(preds, refs, iou_thresholds=[0.3])["metrics"]["all"]
        assert (res["tp"], res["fp"], res["fn"]) == (1, 0, 0)

    def test_no_match_when_threshold_above_iou(self):
        preds, refs = _convert(
            [[_gt(NONSQUARE_GT)]], [[_pred(NONSQUARE_PRED)]], 100, 400
        )
        res = _compute(preds, refs, iou_thresholds=[0.35])["metrics"]["all"]
        assert (res["tp"], res["fp"], res["fn"]) == (0, 1, 1)

    def test_swapping_width_and_height_changes_the_verdict(self):
        """The same relative boxes on a 400x100 image have a different IoU.

        px: gt [0, 0, 200, 50], pred [0, 25, 200, 50] -> intersection 200*25=5000,
        union 20000-5000=15000 -> still 1/3, but the *boxes* differ, so a
        transposed resolution would silently produce different geometry.
        """
        wide_preds, wide_refs = _convert(
            [[_gt(NONSQUARE_GT)]], [[_pred(NONSQUARE_PRED)]], 400, 100
        )
        np.testing.assert_allclose(wide_refs[0]["boxes"], [[0.0, 0.0, 200.0, 50.0]])
        np.testing.assert_allclose(wide_preds[0]["boxes"], [[0.0, 25.0, 200.0, 50.0]])

        tall_preds, tall_refs = _convert(
            [[_gt(NONSQUARE_GT)]], [[_pred(NONSQUARE_PRED)]], 100, 400
        )
        np.testing.assert_allclose(tall_refs[0]["boxes"], [[0.0, 0.0, 50.0, 200.0]])
        np.testing.assert_allclose(tall_preds[0]["boxes"], [[0.0, 100.0, 50.0, 200.0]])


class TestAreaBucketing:
    """The bbox area alone decides which area range a detection lands in.

    A 20x20 px box has an area of 400 px². With area ranges split at 300 it must
    land in "large" — and it must do so whether or not the annotation carries an
    ``area`` attribute, because the metric always derives area from the geometry
    it also uses for the IoU.
    """

    def _run(self, gt_det):
        preds, refs = _convert([[gt_det]], [[_pred(GT_BOX)]], 100, 100)
        return _compute(
            preds,
            refs,
            iou_thresholds=[0.5],
            area_ranges=AREA_RANGES,
            area_ranges_labels=AREA_LABELS,
        )["metrics"]

    def test_missing_area_lands_in_large_bucket(self):
        metrics = self._run(_gt(GT_BOX))
        assert metrics["large"]["tp"] == 1
        assert metrics["large"]["support"] == 1
        assert metrics["small"]["tp"] == 0
        assert metrics["small"]["support"] == 0

    def test_annotated_area_does_not_change_the_bucket(self):
        """An explicit small ``area`` must NOT move the box into "small"."""
        metrics = self._run(_gt(GT_BOX, area=100.0))
        assert metrics["large"]["tp"] == 1
        assert metrics["large"]["support"] == 1
        assert metrics["small"]["tp"] == 0
        assert metrics["small"]["support"] == 0

    def test_area_supplied_directly_to_the_metric_is_ignored(self):
        """Hand-built target dicts cannot override the area either."""
        preds, refs = _convert([[_gt(GT_BOX)]], [[_pred(GT_BOX)]], 100, 100)
        refs[0]["area"] = np.array([100.0])  # would once have said "small"
        metrics = _compute(
            preds,
            refs,
            iou_thresholds=[0.5],
            area_ranges=AREA_RANGES,
            area_ranges_labels=AREA_LABELS,
        )["metrics"]
        assert metrics["large"]["tp"] == 1
        assert metrics["small"]["tp"] == 0

    def test_mismatched_area_length_no_longer_crashes(self):
        """A shorter ``area`` than ``boxes`` used to raise IndexError in compute()."""
        preds, refs = _convert(
            [[_gt(GT_BOX), _gt([0.6, 0.6, 0.2, 0.2])]],
            [[_pred(GT_BOX), _pred([0.6, 0.6, 0.2, 0.2])]],
            100,
            100,
        )
        refs[0]["area"] = np.array([400.0])  # 2 boxes, 1 area
        res = _compute(preds, refs, iou_thresholds=[0.5])["metrics"]["all"]
        assert (res["tp"], res["fp"], res["fn"]) == (2, 0, 0)


class TestEmptyAndMissedFrames:
    """Empty-frame bookkeeping through the fiftyone converter."""

    def test_prediction_on_empty_ground_truth_is_a_false_positive_image(self):
        preds, refs = _convert([[]], [[_pred(PRED_BOX)]], 100, 100)
        res = _compute(preds, refs, iou_thresholds=[0.5])["metrics"]["all"]
        assert (res["tp"], res["fp"], res["fn"]) == (0, 1, 0)
        assert res["fpi"] == 1
        assert res["recall"] == -1  # undefined: no ground truth
        assert res["nImgs"] == 1

    def test_missed_ground_truth_is_a_false_negative(self):
        preds, refs = _convert([[_gt(GT_BOX, area=400.0)]], [[]], 100, 100)
        res = _compute(preds, refs, iou_thresholds=[0.5])["metrics"]["all"]
        assert (res["tp"], res["fp"], res["fn"]) == (0, 0, 1)
        assert res["fpi"] == 0
        assert res["precision"] == -1  # undefined: no predictions
        assert res["support"] == 1

    def test_both_sides_empty(self):
        preds, refs = _convert([[]], [[]], 100, 100)
        res = _compute(preds, refs, iou_thresholds=[0.5])["metrics"]["all"]
        assert (res["tp"], res["fp"], res["fn"]) == (0, 0, 0)
        assert res["fpi"] == 0
        assert res["nImgs"] == 1

    def test_two_frames_one_hit_one_miss(self):
        """1 TP + 1 FN -> precision 1.0, recall 0.5, f1 2/3."""
        gt_frames = [[_gt(GT_BOX, area=400.0)], [_gt(GT_BOX, area=400.0)]]
        pred_frames = [[_pred(GT_BOX)], []]
        preds, refs = _convert(gt_frames, pred_frames, 100, 100)
        res = _compute(preds, refs, iou_thresholds=[0.5])["metrics"]["all"]
        assert (res["tp"], res["fp"], res["fn"]) == (1, 0, 1)
        assert res["precision"] == pytest.approx(1.0)
        assert res["recall"] == pytest.approx(0.5)
        assert res["f1"] == pytest.approx(2 / 3)
        assert res["support"] == 2
        assert res["nImgs"] == 2


class TestPayloadEndToEnd:
    """A payload whose sequences have different resolutions must still evaluate.

    Both sequences hold the same *relative* boxes, so each sequence contributes
    one true positive only if it was scaled with its own resolution.
    """

    def test_mixed_resolution_payload_counts_both_sequences(self):
        sequences = {}
        for name, (width, height) in {"seq_a": (100, 100), "seq_b": (640, 512)}.items():
            sequences[name] = Sequence(
                resolution=Resolution(height=height, width=width),
                gt=[[_gt(GT_BOX, area=None)]],
                model=[[_pred(GT_BOX)]],
            )
        payload = Payload(
            dataset="synthetic",
            models=["model"],
            gt_field_name="gt",
            sequences=sequences,
        )

        preds, refs = payload_to_det_metric(payload)
        # seq_a: 20x20 px box; seq_b: 128x102.4 px box
        np.testing.assert_allclose(refs[0]["boxes"], [[10.0, 10.0, 20.0, 20.0]])
        np.testing.assert_allclose(refs[1]["boxes"], [[64.0, 51.2, 128.0, 102.4]])

        res = _compute(preds, refs, iou_thresholds=[0.5])["metrics"]["all"]
        assert (res["tp"], res["fp"], res["fn"]) == (2, 0, 0)
        assert res["support"] == 2
        assert res["nImgs"] == 2


def _payload(sequences, models=("model",), gt_field="gt"):
    """Wrap a ``{name: Sequence}`` mapping in a Payload."""
    return Payload(
        dataset="synthetic",
        models=list(models),
        gt_field_name=gt_field,
        sequences=sequences,
    )


def _sequence(
    gt_frames,
    pred_frames,
    width=100,
    height=100,
    gt_field="gt",
    model="model",
    keyframes=None,
):
    """Build a single-model Sequence with the given per-frame detection lists."""
    fields = {gt_field: gt_frames, model: pred_frames}
    if keyframes is not None:
        fields["keyframes"] = {model: keyframes}
    return Sequence(resolution=Resolution(height=height, width=width), **fields)


# One sequence where the prediction hits, one where it is missed entirely.
HIT_AND_MISS = {
    "seq_hit": _sequence([[_gt(GT_BOX)]], [[_pred(GT_BOX)]]),
    "seq_miss": _sequence([[_gt(GT_BOX)]], [[]]),
}


class TestPayloadToDetMetricsBySequence:
    """Per-sequence evaluation on the Payload route."""

    def test_one_result_per_sequence_keyed_by_name(self):
        res = payload_to_det_metrics_by_sequence(
            _payload(HIT_AND_MISS), iou_thresholds=[0.5], include_overall=False
        )
        assert list(res) == ["seq_hit", "seq_miss"]

    def test_sequences_are_scored_independently(self):
        """seq_hit is a clean TP; seq_miss is a clean FN. Neither leaks."""
        res = payload_to_det_metrics_by_sequence(
            _payload(HIT_AND_MISS), iou_thresholds=[0.5]
        )
        hit = res["seq_hit"]["metrics"]["all"]
        miss = res["seq_miss"]["metrics"]["all"]
        assert (hit["tp"], hit["fp"], hit["fn"]) == (1, 0, 0)
        assert hit["recall"] == pytest.approx(1.0)
        assert (miss["tp"], miss["fp"], miss["fn"]) == (0, 0, 1)
        assert miss["recall"] == pytest.approx(0.0)
        assert miss["precision"] == -1  # undefined: no predictions in this sequence
        assert hit["nImgs"] == miss["nImgs"] == 1

    def test_per_sequence_reveals_what_pooling_hides(self):
        """Pooled recall 0.5 hides that one sequence scored 1.0 and one 0.0."""
        payload = _payload(HIT_AND_MISS)

        preds, refs = payload_to_det_metric(payload)
        pooled = _compute(preds, refs, iou_thresholds=[0.5])["metrics"]["all"]
        assert (pooled["tp"], pooled["fp"], pooled["fn"]) == (1, 0, 1)
        assert pooled["recall"] == pytest.approx(0.5)

        by_seq = payload_to_det_metrics_by_sequence(payload, iou_thresholds=[0.5])
        recalls = [
            by_seq[k]["metrics"]["all"]["recall"] for k in ("seq_hit", "seq_miss")
        ]
        assert recalls == [pytest.approx(1.0), pytest.approx(0.0)]

    def test_each_sequence_uses_its_own_resolution(self):
        """Same relative boxes, different resolutions -> both still match."""
        sequences = {
            "small": _sequence(
                [[_gt(GT_BOX)]], [[_pred(GT_BOX)]], width=100, height=100
            ),
            "large": _sequence(
                [[_gt(GT_BOX)]], [[_pred(GT_BOX)]], width=640, height=512
            ),
        }
        res = payload_to_det_metrics_by_sequence(
            _payload(sequences), iou_thresholds=[0.5]
        )
        for name in ("small", "large"):
            assert res[name]["metrics"]["all"]["tp"] == 1

    def test_iou_threshold_forwarded_to_every_sequence(self):
        """The IoU=1/3 pair matches at 0.30 and not at 0.35, per sequence."""
        sequences = {"seq": _sequence([[_gt(GT_BOX)]], [[_pred(PRED_BOX)]])}
        loose = payload_to_det_metrics_by_sequence(
            _payload(sequences), iou_thresholds=[0.3]
        )
        strict = payload_to_det_metrics_by_sequence(
            _payload(sequences), iou_thresholds=[0.35]
        )
        assert loose["seq"]["metrics"]["all"]["tp"] == 1
        assert strict["seq"]["metrics"]["all"]["tp"] == 0

    def test_area_ranges_forwarded_to_every_sequence(self):
        res = payload_to_det_metrics_by_sequence(
            _payload(HIT_AND_MISS),
            iou_thresholds=[0.5],
            area_ranges=AREA_RANGES,
            area_ranges_labels=AREA_LABELS,
        )
        for name in ("seq_hit", "seq_miss", OVERALL_KEY):
            assert set(res[name]["metrics"]) == {"small", "large"}
        # the 20x20 px (400 px^2) box belongs to "large"
        assert res["seq_hit"]["metrics"]["large"]["tp"] == 1
        assert res["seq_hit"]["metrics"]["small"]["tp"] == 0

    def test_default_model_is_first_in_payload(self):
        sequences = {
            "seq": Sequence(
                resolution=Resolution(height=100, width=100),
                gt=[[_gt(GT_BOX)]],
                model_a=[[_pred(GT_BOX)]],
                model_b=[[]],
            )
        }
        res = payload_to_det_metrics_by_sequence(
            _payload(sequences, models=("model_a", "model_b")), iou_thresholds=[0.5]
        )
        assert res["seq"]["metrics"]["all"]["tp"] == 1

    def test_explicit_model_name_honoured(self):
        sequences = {
            "seq": Sequence(
                resolution=Resolution(height=100, width=100),
                gt=[[_gt(GT_BOX)]],
                model_a=[[_pred(GT_BOX)]],
                model_b=[[]],
            )
        }
        res = payload_to_det_metrics_by_sequence(
            _payload(sequences, models=("model_a", "model_b")),
            model_name="model_b",
            iou_thresholds=[0.5],
        )
        assert res["seq"]["metrics"]["all"]["fn"] == 1

    def test_ground_truth_read_from_gt_field_name(self):
        sequences = {
            "seq": _sequence(
                [[_gt(GT_BOX)]], [[_pred(GT_BOX)]], gt_field="ground_truth"
            )
        }
        res = payload_to_det_metrics_by_sequence(
            _payload(sequences, gt_field="ground_truth"), iou_thresholds=[0.5]
        )
        assert res["seq"]["metrics"]["all"]["support"] == 1

    def test_sequence_with_no_frames_reports_zero_images(self):
        sequences = {"empty": _sequence([], [])}
        res = payload_to_det_metrics_by_sequence(
            _payload(sequences), iou_thresholds=[0.5]
        )
        metrics = res["empty"]["metrics"]["all"]
        assert metrics["nImgs"] == 0
        assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (0, 0, 0)

    def test_payload_with_no_sequences_returns_empty_dict(self):
        """No sequences means nothing to pool, so no OVERALL entry either."""
        assert payload_to_det_metrics_by_sequence(_payload({})) == {}

    def test_box_format_cannot_be_overridden(self):
        payload = _payload(HIT_AND_MISS)
        with pytest.raises(ValueError, match="box_format` cannot be overridden"):
            payload_to_det_metrics_by_sequence(payload, box_format="xyxy")

    def test_label_mapping_with_class_agnostic_raises(self):
        payload = _payload(HIT_AND_MISS)
        with pytest.raises(ValueError, match="Label mapping cannot be provided"):
            payload_to_det_metrics_by_sequence(payload, label_mapping={"BOAT": 1})


LABEL_MAPPING = {"BOAT": 1, "BUOY": 2}


class TestBySequenceClassSpecific:
    """Class-specific runs must report the same classes for every sequence.

    Without a shared ``labels`` list each sequence would derive its own category
    ids from the labels it happens to contain, so the per-sequence result arrays
    would have different lengths and could not be compared column-wise.
    """

    def _payload_with_disjoint_classes(self):
        boat = fo.Detection(label="BOAT", bounding_box=GT_BOX)
        buoy = fo.Detection(label="BUOY", bounding_box=GT_BOX)
        return _payload(
            {
                "only_boat": _sequence(
                    [[boat]],
                    [[fo.Detection(label="BOAT", bounding_box=GT_BOX, confidence=0.9)]],
                ),
                "only_buoy": _sequence(
                    [[buoy]],
                    [[fo.Detection(label="BUOY", bounding_box=GT_BOX, confidence=0.9)]],
                ),
            }
        )

    def test_labels_default_to_sorted_mapping_values(self):
        res = payload_to_det_metrics_by_sequence(
            self._payload_with_disjoint_classes(),
            label_mapping=LABEL_MAPPING,
            class_agnostic=False,
            iou_thresholds=[0.5],
        )
        for name in ("only_boat", "only_buoy"):
            tp = np.asarray(res[name]["metrics"]["all"]["tp"])
            assert tp.shape == (2,), (
                f"{name} reported {tp.shape}, expected both classes"
            )

        # class 1 (BOAT) hits in only_boat, class 2 (BUOY) hits in only_buoy
        np.testing.assert_array_equal(res["only_boat"]["metrics"]["all"]["tp"], [1, 0])
        np.testing.assert_array_equal(res["only_buoy"]["metrics"]["all"]["tp"], [0, 1])

    def test_explicit_labels_are_not_overridden(self):
        res = payload_to_det_metrics_by_sequence(
            self._payload_with_disjoint_classes(),
            label_mapping=LABEL_MAPPING,
            class_agnostic=False,
            labels=[1, 2, 3],
            iou_thresholds=[0.5],
        )
        tp = np.asarray(res["only_boat"]["metrics"]["all"]["tp"])
        assert tp.shape == (3,)


class TestBySequenceToDataFrame:
    """The result shape must feed sequence_results_to_df unchanged."""

    def test_one_row_per_sequence_plus_overall_last(self):
        res = payload_to_det_metrics_by_sequence(
            _payload(HIT_AND_MISS), iou_thresholds=[0.5]
        )
        df = sequence_results_to_df(res)
        assert list(df["sequence"]) == ["seq_hit", "seq_miss", OVERALL_KEY]
        assert list(df["tp"]) == [1, 0, 1]
        assert list(df["fn"]) == [0, 1, 1]
        assert list(df["n_imgs"]) == [1, 1, 2]

    def test_one_row_per_sequence_and_area_range(self):
        res = payload_to_det_metrics_by_sequence(
            _payload(HIT_AND_MISS),
            iou_thresholds=[0.5],
            area_ranges=AREA_RANGES,
            area_ranges_labels=AREA_LABELS,
        )
        df = sequence_results_to_df(res)
        assert len(df) == 6  # (2 sequences + OVERALL) x 2 area ranges
        assert set(df["area_range_lbl"]) == {"small", "large"}

    def test_overall_omitted_when_disabled(self):
        res = payload_to_det_metrics_by_sequence(
            _payload(HIT_AND_MISS), iou_thresholds=[0.5], include_overall=False
        )
        df = sequence_results_to_df(res)
        assert OVERALL_KEY not in list(df["sequence"])


def _mixed_payload():
    """Two sequences covering TP, FP, FN and a false-positive image.

    seq_a: frame 1 is a clean match; frame 2 has no GT but one prediction
           -> tp 1, fp 1, fpi 1
    seq_b: frame 1 misses its GT; frame 2 has a badly placed prediction
           -> fp 1, fn 2
    """
    return _payload(
        {
            "seq_a": _sequence([[_gt(GT_BOX)], []], [[_pred(GT_BOX)], [_pred(GT_BOX)]]),
            "seq_b": _sequence([[_gt(GT_BOX)], [_gt(GT_BOX)]], [[], [_pred(PRED_BOX)]]),
        }
    )


class TestOverallIsPooledNotAveraged:
    """OVERALL must sum the counts and recompute the ratios once."""

    def test_overall_is_the_last_key(self):
        res = payload_to_det_metrics_by_sequence(
            _payload(HIT_AND_MISS), iou_thresholds=[0.5]
        )
        assert list(res) == ["seq_hit", "seq_miss", OVERALL_KEY]

    def test_counts_are_the_sum_of_the_sequences(self):
        res = payload_to_det_metrics_by_sequence(_mixed_payload(), iou_thresholds=[0.5])
        overall = res[OVERALL_KEY]["metrics"]["all"]
        for key in ("tp", "fp", "fn", "duplicates", "fpi", "support", "nImgs"):
            expected = sum(
                res[name]["metrics"]["all"][key] for name in ("seq_a", "seq_b")
            )
            assert overall[key] == expected, key
        # hand-checked: seq_a (1,1,0) + seq_b (0,1,2)
        assert (overall["tp"], overall["fp"], overall["fn"]) == (1, 2, 2)
        assert overall["fpi"] == 1
        assert overall["nImgs"] == 4

    def test_ratios_recomputed_from_pooled_counts(self):
        """1/(1+2) = 1/3 for both precision and recall, so f1 is also 1/3."""
        res = payload_to_det_metrics_by_sequence(_mixed_payload(), iou_thresholds=[0.5])
        overall = res[OVERALL_KEY]["metrics"]["all"]
        assert overall["precision"] == pytest.approx(1 / 3)
        assert overall["recall"] == pytest.approx(1 / 3)
        assert overall["f1"] == pytest.approx(1 / 3)

    def test_overall_is_not_the_mean_of_sequence_ratios(self):
        """seq_a precision 0.5 and seq_b 0.0 average to 0.25, not 1/3."""
        res = payload_to_det_metrics_by_sequence(_mixed_payload(), iou_thresholds=[0.5])
        per_seq = [
            res[name]["metrics"]["all"]["precision"] for name in ("seq_a", "seq_b")
        ]
        assert per_seq == [pytest.approx(0.5), pytest.approx(0.0)]
        assert res[OVERALL_KEY]["metrics"]["all"]["precision"] != pytest.approx(0.25)

    def test_overall_matches_a_genuinely_pooled_metric_exactly(self):
        """The whole point: no second pass, but identical to one big evaluation."""
        payload = _mixed_payload()
        preds, refs = payload_to_det_metric(payload)
        pooled = _compute(preds, refs, iou_thresholds=[0.5])["metrics"]["all"]

        overall = payload_to_det_metrics_by_sequence(payload, iou_thresholds=[0.5])[
            OVERALL_KEY
        ]["metrics"]["all"]
        assert overall == pooled

    def test_overall_matches_pooled_metric_per_area_range(self):
        payload = _mixed_payload()
        preds, refs = payload_to_det_metric(payload)
        pooled = _compute(
            preds,
            refs,
            iou_thresholds=[0.5],
            area_ranges=AREA_RANGES,
            area_ranges_labels=AREA_LABELS,
        )["metrics"]

        overall = payload_to_det_metrics_by_sequence(
            payload,
            iou_thresholds=[0.5],
            area_ranges=AREA_RANGES,
            area_ranges_labels=AREA_LABELS,
        )[OVERALL_KEY]["metrics"]
        assert set(overall) == set(pooled)
        for area_label in pooled:
            assert overall[area_label] == pooled[area_label], area_label


class TestOverallSentinels:
    """Undefined pooled ratios must use the same -1 sentinel as COCOeval."""

    def test_no_predictions_anywhere_gives_undefined_precision(self):
        payload = _payload(
            {
                "a": _sequence([[_gt(GT_BOX)]], [[]]),
                "b": _sequence([[_gt(GT_BOX)]], [[]]),
            }
        )
        overall = payload_to_det_metrics_by_sequence(payload, iou_thresholds=[0.5])[
            OVERALL_KEY
        ]["metrics"]["all"]
        assert (overall["tp"], overall["fp"], overall["fn"]) == (0, 0, 2)
        assert overall["precision"] == -1
        assert overall["recall"] == pytest.approx(0.0)
        assert overall["f1"] == -1

    def test_no_ground_truth_anywhere_gives_undefined_recall(self):
        payload = _payload(
            {
                "a": _sequence([[]], [[_pred(GT_BOX)]]),
                "b": _sequence([[]], [[_pred(GT_BOX)]]),
            }
        )
        overall = payload_to_det_metrics_by_sequence(payload, iou_thresholds=[0.5])[
            OVERALL_KEY
        ]["metrics"]["all"]
        assert (overall["tp"], overall["fp"], overall["fn"]) == (0, 2, 0)
        assert overall["precision"] == pytest.approx(0.0)
        assert overall["recall"] == -1
        assert overall["f1"] == -1
        assert overall["fpi"] == 2

    def test_completely_empty_sequences_give_all_undefined(self):
        payload = _payload({"a": _sequence([[]], [[]]), "b": _sequence([[]], [[]])})
        overall = payload_to_det_metrics_by_sequence(payload, iou_thresholds=[0.5])[
            OVERALL_KEY
        ]["metrics"]["all"]
        assert (overall["tp"], overall["fp"], overall["fn"]) == (0, 0, 0)
        assert overall["precision"] == overall["recall"] == overall["f1"] == -1
        assert overall["nImgs"] == 2

    def test_scalar_types_match_cocoeval(self):
        """Class-agnostic mode returns python ints and floats, not numpy scalars."""
        overall = payload_to_det_metrics_by_sequence(
            _payload(HIT_AND_MISS), iou_thresholds=[0.5]
        )[OVERALL_KEY]["metrics"]["all"]
        for key in ("tp", "fp", "fn", "duplicates", "support", "fpi", "nImgs"):
            assert type(overall[key]) is int, key
        for key in ("precision", "recall", "f1"):
            assert type(overall[key]) is float, key


class TestOverallClassSpecific:
    """Class-specific pooling sums per-class counts position by position."""

    def _payload_with_disjoint_classes(self):
        return _payload(
            {
                "only_boat": _sequence(
                    [[fo.Detection(label="BOAT", bounding_box=GT_BOX)]],
                    [[fo.Detection(label="BOAT", bounding_box=GT_BOX, confidence=0.9)]],
                ),
                "only_buoy": _sequence(
                    [[fo.Detection(label="BUOY", bounding_box=GT_BOX)]],
                    [[fo.Detection(label="BUOY", bounding_box=GT_BOX, confidence=0.9)]],
                ),
            }
        )

    def test_per_class_counts_are_summed(self):
        res = payload_to_det_metrics_by_sequence(
            self._payload_with_disjoint_classes(),
            label_mapping=LABEL_MAPPING,
            class_agnostic=False,
            iou_thresholds=[0.5],
        )
        # BOAT hits in only_boat -> [1, 0]; BUOY hits in only_buoy -> [0, 1]
        np.testing.assert_array_equal(res["only_boat"]["metrics"]["all"]["tp"], [1, 0])
        np.testing.assert_array_equal(res["only_buoy"]["metrics"]["all"]["tp"], [0, 1])
        overall = res[OVERALL_KEY]["metrics"]["all"]
        np.testing.assert_array_equal(overall["tp"], [1, 1])
        np.testing.assert_array_equal(overall["fn"], [0, 0])
        np.testing.assert_array_equal(overall["precision"], [1.0, 1.0])

    def test_overall_matches_pooled_metric_per_class(self):
        payload = self._payload_with_disjoint_classes()
        preds, refs = payload_to_det_metric(
            payload, label_mapping=LABEL_MAPPING, class_agnostic=False
        )
        pooled = _compute(
            preds,
            refs,
            iou_thresholds=[0.5],
            class_agnostic=False,
            labels=sorted(LABEL_MAPPING.values()),
        )["metrics"]["all"]
        overall = payload_to_det_metrics_by_sequence(
            payload,
            label_mapping=LABEL_MAPPING,
            class_agnostic=False,
            iou_thresholds=[0.5],
        )[OVERALL_KEY]["metrics"]["all"]
        for key in ("tp", "fp", "fn", "precision", "recall", "f1", "support"):
            np.testing.assert_allclose(overall[key], pooled[key], err_msg=key)


class TestAggregateSequenceResults:
    """Direct tests for the pooling helper."""

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="No sequence results to pool"):
            aggregate_sequence_results({})

    def test_only_overall_entry_raises(self):
        res = payload_to_det_metrics_by_sequence(
            _payload(HIT_AND_MISS), iou_thresholds=[0.5]
        )
        with pytest.raises(ValueError, match="No sequence results to pool"):
            aggregate_sequence_results({OVERALL_KEY: res[OVERALL_KEY]})

    def test_existing_overall_is_skipped_so_pooling_is_idempotent(self):
        res = payload_to_det_metrics_by_sequence(_mixed_payload(), iou_thresholds=[0.5])
        again = aggregate_sequence_results(res)
        assert again["metrics"]["all"] == res[OVERALL_KEY]["metrics"]["all"]

    def test_mismatched_area_labels_raise(self):
        one = payload_to_det_metrics_by_sequence(
            _payload({"a": _sequence([[_gt(GT_BOX)]], [[_pred(GT_BOX)]])}),
            iou_thresholds=[0.5],
            include_overall=False,
        )
        two = payload_to_det_metrics_by_sequence(
            _payload({"b": _sequence([[_gt(GT_BOX)]], [[_pred(GT_BOX)]])}),
            iou_thresholds=[0.5],
            area_ranges=AREA_RANGES,
            area_ranges_labels=AREA_LABELS,
            include_overall=False,
        )
        with pytest.raises(ValueError, match="disagree on area-range labels"):
            aggregate_sequence_results({**one, **two})

    def test_mismatched_class_shapes_raise(self):
        """Per-sequence runs without a shared `labels` list cannot be pooled."""
        boat = payload_to_det_metrics_by_sequence(
            _payload({"a": _sequence([[_gt(GT_BOX)]], [[_pred(GT_BOX)]])}),
            iou_thresholds=[0.5],
            include_overall=False,
        )
        multi = payload_to_det_metrics_by_sequence(
            _payload({"b": _sequence([[_gt(GT_BOX)]], [[_pred(GT_BOX)]])}),
            label_mapping=LABEL_MAPPING,
            class_agnostic=False,
            iou_thresholds=[0.5],
            include_overall=False,
        )
        with pytest.raises(ValueError, match="Inconsistent `tp` shapes"):
            aggregate_sequence_results({**boat, **multi})

    def test_result_carries_only_the_metrics_key(self):
        res = payload_to_det_metrics_by_sequence(
            _payload(HIT_AND_MISS), iou_thresholds=[0.5], include_overall=False
        )
        assert set(aggregate_sequence_results(res)) == {"metrics"}


class TestOverallKeyCollision:
    """A real sequence named OVERALL must not be silently overwritten."""

    def test_sequence_named_overall_raises(self):
        payload = _payload({OVERALL_KEY: _sequence([[_gt(GT_BOX)]], [[_pred(GT_BOX)]])})
        with pytest.raises(ValueError, match="collides with the pooled entry"):
            payload_to_det_metrics_by_sequence(payload, iou_thresholds=[0.5])

    def test_sequence_named_overall_allowed_when_disabled(self):
        payload = _payload({OVERALL_KEY: _sequence([[_gt(GT_BOX)]], [[_pred(GT_BOX)]])})
        res = payload_to_det_metrics_by_sequence(
            payload, iou_thresholds=[0.5], include_overall=False
        )
        assert res[OVERALL_KEY]["metrics"]["all"]["tp"] == 1


# Ground truth annotated on all four frames; the model only emits on frames 0
# and 2, which are the frames it flagged as keyframes.
DENSE_GT = [[_gt(GT_BOX)], [_gt(GT_BOX)], [_gt(GT_BOX)], [_gt(GT_BOX)]]
SPARSE_PRED = [[_pred(GT_BOX)], [], [_pred(GT_BOX)], []]
KEYFRAME_MASK = [True, False, True, False]


def _keyframe_payload(mask=KEYFRAME_MASK, gt=None, pred=None):
    """Payload whose single sequence carries a keyframe mask for "model"."""
    return _payload(
        {
            "seq": _sequence(
                DENSE_GT if gt is None else gt,
                SPARSE_PRED if pred is None else pred,
                keyframes=mask,
            )
        }
    )


class TestKeyframesOnly:
    """Dropping non-keyframe frames, mirroring seametrics.tracking."""

    def test_disabled_by_default_every_frame_counts(self):
        """Frames the model skipped are misses: recall 2/4, nImgs 4."""
        res = payload_to_det_metrics_by_sequence(
            _keyframe_payload(), iou_thresholds=[0.5]
        )["seq"]["metrics"]["all"]
        assert (res["tp"], res["fp"], res["fn"]) == (2, 0, 2)
        assert res["recall"] == pytest.approx(0.5)
        assert res["nImgs"] == 4

    def test_enabled_drops_non_keyframes_from_both_sides(self):
        """Only frames 0 and 2 survive: recall 2/2, nImgs 2."""
        res = payload_to_det_metrics_by_sequence(
            _keyframe_payload(), iou_thresholds=[0.5], keyframes_only=True
        )["seq"]["metrics"]["all"]
        assert (res["tp"], res["fp"], res["fn"]) == (2, 0, 0)
        assert res["recall"] == pytest.approx(1.0)
        assert res["nImgs"] == 2

    def test_ground_truth_on_dropped_frames_is_not_a_miss(self):
        """The dropped frames held real GT; it must not resurface as fn."""
        res = payload_to_det_metrics_by_sequence(
            _keyframe_payload(), iou_thresholds=[0.5], keyframes_only=True
        )["seq"]["metrics"]["all"]
        assert res["fn"] == 0
        assert res["support"] == 2  # 2 keyframes, one GT object each

    def test_predictions_on_dropped_frames_are_not_false_positives(self):
        """A stray prediction on a non-keyframe must not count against precision."""
        pred = [[_pred(GT_BOX)], [_pred(PRED_BOX)], [_pred(GT_BOX)], []]
        res = payload_to_det_metrics_by_sequence(
            _keyframe_payload(pred=pred), iou_thresholds=[0.5], keyframes_only=True
        )["seq"]["metrics"]["all"]
        assert (res["tp"], res["fp"], res["fn"]) == (2, 0, 0)
        assert res["precision"] == pytest.approx(1.0)

    def test_all_frames_flagged_is_a_no_op(self):
        mask = [True, True, True, True]
        on = payload_to_det_metrics_by_sequence(
            _keyframe_payload(mask=mask), iou_thresholds=[0.5], keyframes_only=True
        )["seq"]["metrics"]["all"]
        off = payload_to_det_metrics_by_sequence(
            _keyframe_payload(mask=mask), iou_thresholds=[0.5]
        )["seq"]["metrics"]["all"]
        assert on == off

    def test_no_frames_flagged_yields_an_empty_evaluation(self):
        res = payload_to_det_metrics_by_sequence(
            _keyframe_payload(mask=[False] * 4),
            iou_thresholds=[0.5],
            keyframes_only=True,
        )["seq"]["metrics"]["all"]
        assert (res["tp"], res["fp"], res["fn"]) == (0, 0, 0)
        assert res["nImgs"] == 0

    def test_missing_mask_raises_rather_than_silently_evaluating_all(self):
        payload = _payload({"seq": _sequence(DENSE_GT, SPARSE_PRED)})  # no keyframes
        with pytest.raises(ValueError, match="no keyframe data"):
            payload_to_det_metrics_by_sequence(
                payload, iou_thresholds=[0.5], keyframes_only=True
            )

    def test_mask_for_a_different_model_raises(self):
        seq = Sequence(
            resolution=Resolution(height=100, width=100),
            gt=DENSE_GT,
            model_a=SPARSE_PRED,
            model_b=SPARSE_PRED,
            keyframes={"model_a": KEYFRAME_MASK},
        )
        payload = _payload({"seq": seq}, models=("model_a", "model_b"))
        with pytest.raises(ValueError, match="no keyframe data"):
            payload_to_det_metrics_by_sequence(
                payload,
                model_name="model_b",
                iou_thresholds=[0.5],
                keyframes_only=True,
            )

    def test_mask_length_mismatch_raises(self):
        payload = _keyframe_payload(mask=[True, False])  # 2 flags, 4 frames
        with pytest.raises(ValueError, match="has 2 entries"):
            payload_to_det_metrics_by_sequence(
                payload, iou_thresholds=[0.5], keyframes_only=True
            )

    def test_overall_pools_the_filtered_counts(self):
        payload = _payload(
            {
                "a": _sequence(DENSE_GT, SPARSE_PRED, keyframes=KEYFRAME_MASK),
                "b": _sequence(DENSE_GT, SPARSE_PRED, keyframes=KEYFRAME_MASK),
            }
        )
        res = payload_to_det_metrics_by_sequence(
            payload, iou_thresholds=[0.5], keyframes_only=True
        )
        overall = res[OVERALL_KEY]["metrics"]["all"]
        assert (overall["tp"], overall["fp"], overall["fn"]) == (4, 0, 0)
        assert overall["nImgs"] == 4  # 2 keyframes x 2 sequences

    def test_pooled_route_agrees_with_by_sequence_route(self):
        payload = _keyframe_payload()
        preds, refs = payload_to_det_metric(payload, keyframes_only=True)
        pooled = _compute(preds, refs, iou_thresholds=[0.5])["metrics"]["all"]
        overall = payload_to_det_metrics_by_sequence(
            payload, iou_thresholds=[0.5], keyframes_only=True
        )[OVERALL_KEY]["metrics"]["all"]
        assert overall == pooled
