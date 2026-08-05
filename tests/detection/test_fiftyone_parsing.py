"""Tests for the fiftyone -> detection-metric conversion in seametrics.detection.utils.

The functions under test translate fiftyone's *relative* ``[x, y, w, h]``
bounding boxes into absolute-pixel ``xywh`` boxes plus labels and scores. Every
expected value below is hand-computed from the box and the resolution in the
test body.
"""

import fiftyone as fo
import numpy as np
import pytest

from seametrics.detection.utils import (
    box_denormalize,
    frame_dets_to_det_metrics,
    payload_sequence_to_det_metrics,
    payload_to_det_metric,
)
from seametrics.payload import Payload, Resolution, Sequence

# ---------------------------------------------------------------------------
# frame_dets_to_det_metrics: fiftyone -> metric box/width/height parsing
# ---------------------------------------------------------------------------


class TestFrameDetsBoxGeometry:
    """Geometry of the fiftyone -> metric box conversion.

    FiftyOne stores ``bounding_box`` as relative ``[x, y, width, height]``. The
    converter must scale x-like values by the image width, y-like values by the
    image height, and keep the ``xywh`` layout (NOT ``xyxy``).
    """

    def test_relative_xywh_scaled_to_pixels(self):
        """[0.25, 0.5, 0.5, 0.25] on 640x512 -> [160, 256, 320, 128]."""
        dets = [
            fo.Detection(label="A", bounding_box=[0.25, 0.5, 0.5, 0.25], confidence=0.5)
        ]
        out = frame_dets_to_det_metrics(dets, 640, 512)
        np.testing.assert_allclose(out["boxes"], [[160.0, 256.0, 320.0, 128.0]])

    def test_output_is_xywh_not_xyxy(self):
        """Third/fourth values must be width/height, not bottom-right corner.

        For bbox [0.5, 0.5, 0.1, 0.2] on 100x200 the xywh answer is
        [50, 100, 10, 40]; an xyxy answer would be [50, 100, 60, 140].
        """
        dets = [
            fo.Detection(label="A", bounding_box=[0.5, 0.5, 0.1, 0.2], confidence=1.0)
        ]
        out = frame_dets_to_det_metrics(dets, 100, 200)
        np.testing.assert_allclose(out["boxes"], [[50.0, 100.0, 10.0, 40.0]])

    def test_width_and_height_are_not_interchangeable(self):
        """Swapping w and h must change the result for a non-square box."""
        dets = [
            fo.Detection(label="A", bounding_box=[0.1, 0.2, 0.3, 0.4], confidence=1.0)
        ]
        wide = frame_dets_to_det_metrics(dets, 400, 100)
        tall = frame_dets_to_det_metrics(dets, 100, 400)
        np.testing.assert_allclose(wide["boxes"], [[40.0, 20.0, 120.0, 40.0]])
        np.testing.assert_allclose(tall["boxes"], [[10.0, 80.0, 30.0, 160.0]])

    def test_full_frame_box_covers_whole_image(self):
        """[0, 0, 1, 1] must map to the full image extent."""
        dets = [fo.Detection(label="A", bounding_box=[0, 0, 1, 1], confidence=1.0)]
        out = frame_dets_to_det_metrics(dets, 640, 512)
        np.testing.assert_allclose(out["boxes"], [[0.0, 0.0, 640.0, 512.0]])

    def test_box_extending_past_right_edge_is_not_clipped(self):
        """Boxes with x + w > 1 are kept as-is; clipping is not this function's job."""
        dets = [
            fo.Detection(label="A", bounding_box=[0.9, 0.9, 0.5, 0.5], confidence=1.0)
        ]
        out = frame_dets_to_det_metrics(dets, 100, 200)
        np.testing.assert_allclose(out["boxes"], [[90.0, 180.0, 50.0, 100.0]])

    def test_zero_sized_box_stays_zero_sized(self):
        dets = [
            fo.Detection(label="A", bounding_box=[0.5, 0.5, 0.0, 0.0], confidence=1.0)
        ]
        out = frame_dets_to_det_metrics(dets, 100, 200)
        np.testing.assert_allclose(out["boxes"], [[50.0, 100.0, 0.0, 0.0]])

    def test_multiple_detections_keep_input_order(self):
        dets = [
            fo.Detection(label="A", bounding_box=[0.0, 0.0, 0.5, 0.5], confidence=0.9),
            fo.Detection(label="A", bounding_box=[0.5, 0.5, 0.5, 0.5], confidence=0.1),
        ]
        out = frame_dets_to_det_metrics(dets, 100, 200)
        np.testing.assert_allclose(
            out["boxes"], [[0.0, 0.0, 50.0, 100.0], [50.0, 100.0, 50.0, 100.0]]
        )
        np.testing.assert_allclose(out["scores"], [0.9, 0.1])


class TestFrameDetsScores:
    """Confidence handling for predictions."""

    def test_zero_confidence_is_preserved(self):
        """Regression: a 0.0 confidence must not be promoted to 1.0."""
        dets = [
            fo.Detection(label="A", bounding_box=[0.1, 0.1, 0.2, 0.2], confidence=0.0)
        ]
        out = frame_dets_to_det_metrics(dets, 100, 100)
        np.testing.assert_array_equal(out["scores"], [0.0])

    def test_missing_confidence_defaults_to_one(self):
        """Ground-truth-style detections carry no confidence -> 1.0."""
        dets = [fo.Detection(label="A", bounding_box=[0.1, 0.1, 0.2, 0.2])]
        out = frame_dets_to_det_metrics(dets, 100, 100)
        np.testing.assert_array_equal(out["scores"], [1.0])

    def test_confidence_forwarded_verbatim(self):
        dets = [
            fo.Detection(
                label="A", bounding_box=[0, 0, 1, 1], confidence=0.153076171875
            )
        ]
        out = frame_dets_to_det_metrics(dets, 640, 512)
        np.testing.assert_array_equal(out["scores"], [0.153076171875])

    def test_mixed_missing_and_present_confidences_stay_aligned(self):
        dets = [
            fo.Detection(label="A", bounding_box=[0.0, 0.0, 0.1, 0.1], confidence=0.0),
            fo.Detection(label="A", bounding_box=[0.1, 0.1, 0.1, 0.1]),
            fo.Detection(label="A", bounding_box=[0.2, 0.2, 0.1, 0.1], confidence=0.75),
        ]
        out = frame_dets_to_det_metrics(dets, 100, 100)
        np.testing.assert_array_equal(out["scores"], [0.0, 1.0, 0.75])
        assert out["boxes"].shape == (3, 4)


class TestFrameDetsNoArea:
    """The converter must not emit an ``area`` key.

    ``PrecisionRecallF1Support`` always derives the area from the bounding box, so
    forwarding a fiftyone ``area`` attribute would create a value that silently
    disagrees with the geometry the metric actually uses.
    """

    def test_ground_truth_has_no_area_key(self):
        dets = [fo.Detection(label="A", bounding_box=[0.1, 0.1, 0.2, 0.2])]
        out = frame_dets_to_det_metrics(dets, 100, 100, is_gt=True)
        assert "area" not in out

    def test_annotated_area_is_ignored(self):
        """An ``area`` attribute on the annotation changes nothing in the output."""
        with_area = frame_dets_to_det_metrics(
            [fo.Detection(label="A", bounding_box=[0.1, 0.1, 0.2, 0.2], area=1234.5)],
            100,
            100,
            is_gt=True,
        )
        without_area = frame_dets_to_det_metrics(
            [fo.Detection(label="A", bounding_box=[0.1, 0.1, 0.2, 0.2])],
            100,
            100,
            is_gt=True,
        )
        assert set(with_area) == set(without_area)
        np.testing.assert_array_equal(with_area["boxes"], without_area["boxes"])

    def test_no_area_warning_is_printed(self, capsys):
        """The old "Area not found" warning is gone; area is never looked up."""
        dets = [fo.Detection(label="A", bounding_box=[0.1, 0.1, 0.2, 0.2])]
        frame_dets_to_det_metrics(dets, 100, 100, is_gt=True)
        assert "Area not found" not in capsys.readouterr().out

    def test_pixel_box_area_recoverable_from_boxes(self):
        """Callers who want the area can read it off the returned boxes."""
        dets = [
            fo.Detection(label="A", bounding_box=[0.0, 0.0, 0.25, 0.5]),
            fo.Detection(label="A", bounding_box=[0.5, 0.5, 0.1, 0.2]),
        ]
        out = frame_dets_to_det_metrics(dets, 640, 512, is_gt=True)
        np.testing.assert_allclose(
            out["boxes"][:, 2] * out["boxes"][:, 3], [160.0 * 256.0, 64.0 * 102.4]
        )


class TestFrameDetsKeysAndShapes:
    """Output keys and array shapes."""

    def test_prediction_keys(self):
        dets = [fo.Detection(label="A", bounding_box=[0, 0, 1, 1], confidence=0.5)]
        out = frame_dets_to_det_metrics(dets, 100, 100)
        assert set(out) == {"boxes", "labels", "scores"}

    def test_ground_truth_keys(self):
        dets = [fo.Detection(label="A", bounding_box=[0, 0, 1, 1], area=1.0)]
        out = frame_dets_to_det_metrics(dets, 100, 100, is_gt=True)
        assert set(out) == {"boxes", "labels"}

    def test_non_empty_frame_boxes_are_2d(self):
        dets = [fo.Detection(label="A", bounding_box=[0, 0, 1, 1], confidence=0.5)]
        out = frame_dets_to_det_metrics(dets, 100, 100)
        assert out["boxes"].shape == (1, 4)
        assert out["labels"].shape == (1,)
        assert out["scores"].shape == (1,)

    def test_empty_frame_yields_1d_empty_arrays(self):
        """Documented contract: empty frames give shape-(0,) arrays, not (0, 4)."""
        out = frame_dets_to_det_metrics([], 100, 100)
        assert out["boxes"].shape == (0,)
        assert out["labels"].shape == (0,)
        assert out["scores"].shape == (0,)

    def test_empty_ground_truth_frame_yields_1d_empty_arrays(self):
        out = frame_dets_to_det_metrics([], 100, 100, is_gt=True)
        assert out["boxes"].shape == (0,)
        assert out["labels"].shape == (0,)
        assert "area" not in out


class TestFrameDetsLabels:
    """Label assignment and label_mapping filtering."""

    def test_class_agnostic_labels_are_all_zero(self):
        dets = [
            fo.Detection(
                label="BOAT", bounding_box=[0.0, 0.0, 0.1, 0.1], confidence=0.5
            ),
            fo.Detection(
                label="BUOY", bounding_box=[0.1, 0.1, 0.1, 0.1], confidence=0.5
            ),
        ]
        out = frame_dets_to_det_metrics(dets, 100, 100)
        np.testing.assert_array_equal(out["labels"], [0, 0])

    def test_label_mapping_applied(self):
        dets = [
            fo.Detection(
                label="BOAT", bounding_box=[0.0, 0.0, 0.1, 0.1], confidence=0.5
            ),
            fo.Detection(
                label="BUOY", bounding_box=[0.1, 0.1, 0.1, 0.1], confidence=0.5
            ),
        ]
        out = frame_dets_to_det_metrics(
            dets, 100, 100, label_mapping={"BOAT": 3, "BUOY": 7}
        )
        np.testing.assert_array_equal(out["labels"], [3, 7])

    def test_unmapped_label_dropped_keeping_score_alignment(self):
        dets = [
            fo.Detection(
                label="BOAT", bounding_box=[0.0, 0.0, 0.1, 0.1], confidence=0.9
            ),
            fo.Detection(
                label="UNKNOWN", bounding_box=[0.1, 0.1, 0.1, 0.1], confidence=0.1
            ),
            fo.Detection(
                label="BOAT", bounding_box=[0.2, 0.2, 0.1, 0.1], confidence=0.5
            ),
        ]
        out = frame_dets_to_det_metrics(dets, 100, 100, label_mapping={"BOAT": 1})
        assert out["boxes"].shape == (2, 4)
        np.testing.assert_array_equal(out["labels"], [1, 1])
        np.testing.assert_array_equal(out["scores"], [0.9, 0.5])
        np.testing.assert_allclose(out["boxes"][:, 0], [0.0, 20.0])

    def test_unmapped_label_dropped_keeping_box_alignment_for_gt(self):
        dets = [
            fo.Detection(label="BOAT", bounding_box=[0.0, 0.0, 0.1, 0.1]),
            fo.Detection(label="UNKNOWN", bounding_box=[0.1, 0.1, 0.1, 0.1]),
            fo.Detection(label="BOAT", bounding_box=[0.2, 0.2, 0.1, 0.1]),
        ]
        out = frame_dets_to_det_metrics(
            dets, 100, 100, is_gt=True, label_mapping={"BOAT": 1}
        )
        np.testing.assert_array_equal(out["labels"], [1, 1])
        np.testing.assert_allclose(out["boxes"][:, 0], [0.0, 20.0])

    def test_all_labels_dropped_yields_empty_arrays(self):
        dets = [
            fo.Detection(
                label="UNKNOWN", bounding_box=[0.0, 0.0, 0.1, 0.1], confidence=0.5
            )
        ]
        out = frame_dets_to_det_metrics(dets, 100, 100, label_mapping={"BOAT": 1})
        assert out["boxes"].shape == (0,)
        assert out["labels"].shape == (0,)
        assert out["scores"].shape == (0,)

    def test_labels_are_integer_dtype(self):
        """`_get_coco_format` rejects non-integer category ids."""
        dets = [
            fo.Detection(
                label="BOAT", bounding_box=[0.0, 0.0, 0.1, 0.1], confidence=0.5
            )
        ]
        out = frame_dets_to_det_metrics(dets, 100, 100, label_mapping={"BOAT": 1})
        assert out["labels"].dtype.kind == "i"


# ---------------------------------------------------------------------------
# payload_sequence_to_det_metrics
# ---------------------------------------------------------------------------


class TestPayloadSequenceToDetMetrics:
    """Per-frame fan-out over a sequence."""

    def test_one_dict_per_frame_including_empty_frames(self):
        sequence_dets = [
            [
                fo.Detection(
                    label="A", bounding_box=[0.0, 0.0, 0.5, 0.5], confidence=0.5
                )
            ],
            [],
            [
                fo.Detection(
                    label="A", bounding_box=[0.1, 0.1, 0.1, 0.1], confidence=0.5
                ),
                fo.Detection(
                    label="A", bounding_box=[0.2, 0.2, 0.1, 0.1], confidence=0.5
                ),
            ],
        ]
        out = payload_sequence_to_det_metrics(sequence_dets, 100, 200)
        assert len(out) == 3
        assert out[0]["boxes"].shape == (1, 4)
        assert out[1]["boxes"].shape == (0,)
        assert out[2]["boxes"].shape == (2, 4)

    def test_frame_order_preserved(self):
        sequence_dets = [
            [
                fo.Detection(
                    label="A", bounding_box=[i / 10, 0.0, 0.1, 0.1], confidence=0.5
                )
            ]
            for i in range(5)
        ]
        out = payload_sequence_to_det_metrics(sequence_dets, 100, 100)
        np.testing.assert_allclose(
            [frame["boxes"][0, 0] for frame in out], [0.0, 10.0, 20.0, 30.0, 40.0]
        )

    def test_resolution_applied_to_every_frame(self):
        sequence_dets = [
            [fo.Detection(label="A", bounding_box=[0.5, 0.5, 0.5, 0.5], confidence=0.5)]
        ] * 3
        out = payload_sequence_to_det_metrics(sequence_dets, 400, 100)
        for frame in out:
            np.testing.assert_allclose(frame["boxes"], [[200.0, 50.0, 200.0, 50.0]])

    def test_is_gt_flag_propagated(self):
        sequence_dets = [
            [fo.Detection(label="A", bounding_box=[0.0, 0.0, 0.1, 0.1], area=3.0)]
        ]
        out = payload_sequence_to_det_metrics(sequence_dets, 100, 100, is_gt=True)
        assert set(out[0]) == {"boxes", "labels"}

    def test_label_mapping_propagated(self):
        sequence_dets = [
            [
                fo.Detection(
                    label="BOAT", bounding_box=[0.0, 0.0, 0.1, 0.1], confidence=0.5
                ),
                fo.Detection(
                    label="UNKNOWN", bounding_box=[0.1, 0.1, 0.1, 0.1], confidence=0.5
                ),
            ]
        ]
        out = payload_sequence_to_det_metrics(
            sequence_dets, 100, 100, label_mapping={"BOAT": 2}
        )
        np.testing.assert_array_equal(out[0]["labels"], [2])


# ---------------------------------------------------------------------------
# payload_to_det_metric
# ---------------------------------------------------------------------------


def _make_payload(models=("model",), gt_field="gt"):
    """Build a two-sequence payload with deliberately different resolutions.

    Both sequences hold a single frame with one detection at the same *relative*
    coordinates, so any resolution mix-up shows up as a wrong pixel box.
    """
    box = [0.5, 0.5, 0.25, 0.25]

    def _frames(confidence):
        return [[fo.Detection(label="BOAT", bounding_box=box, confidence=confidence)]]

    def _gt_frames():
        return [[fo.Detection(label="BOAT", bounding_box=box)]]

    sequences = {}
    for name, (width, height) in {"seq_a": (200, 100), "seq_b": (400, 50)}.items():
        fields = {gt_field: _gt_frames()}
        for i, model in enumerate(models):
            fields[model] = _frames(0.5 + 0.1 * i)
        sequences[name] = Sequence(
            resolution=Resolution(height=height, width=width), **fields
        )

    return Payload(
        dataset="synthetic",
        models=list(models),
        gt_field_name=gt_field,
        sequences=sequences,
    )


class TestPayloadToDetMetric:
    """Payload-level conversion, including per-sequence resolution."""

    def test_each_sequence_scaled_by_its_own_resolution(self):
        """seq_a is 200x100 and seq_b is 400x50 for the same relative box."""
        preds, refs = payload_to_det_metric(_make_payload())
        np.testing.assert_allclose(preds[0]["boxes"], [[100.0, 50.0, 50.0, 25.0]])
        np.testing.assert_allclose(preds[1]["boxes"], [[200.0, 25.0, 100.0, 12.5]])
        np.testing.assert_allclose(refs[0]["boxes"], preds[0]["boxes"])
        np.testing.assert_allclose(refs[1]["boxes"], preds[1]["boxes"])

    def test_predictions_and_references_have_equal_length(self):
        preds, refs = payload_to_det_metric(_make_payload())
        assert len(preds) == len(refs) == 2

    def test_default_model_is_first_in_payload(self):
        payload = _make_payload(models=("model_a", "model_b"))
        preds, _ = payload_to_det_metric(payload)
        np.testing.assert_allclose(preds[0]["scores"], [0.5])

    def test_explicit_model_name_honoured(self):
        payload = _make_payload(models=("model_a", "model_b"))
        preds, _ = payload_to_det_metric(payload, model_name="model_b")
        np.testing.assert_allclose(preds[0]["scores"], [0.6])

    def test_ground_truth_read_from_gt_field_name(self):
        """A non-default gt field name must still resolve (seq_a is 200x100)."""
        payload = _make_payload(gt_field="ground_truth")
        _, refs = payload_to_det_metric(payload)
        assert len(refs) == 2
        np.testing.assert_allclose(refs[0]["boxes"], [[100.0, 50.0, 50.0, 25.0]])

    def test_predictions_carry_scores_references_do_not(self):
        preds, refs = payload_to_det_metric(_make_payload())
        assert set(preds[0]) == {"boxes", "labels", "scores"}
        assert set(refs[0]) == {"boxes", "labels"}

    def test_label_mapping_with_class_agnostic_raises(self):
        with pytest.raises(ValueError, match="Label mapping cannot be provided"):
            payload_to_det_metric(
                _make_payload(), label_mapping={"BOAT": 1}, class_agnostic=True
            )

    def test_label_mapping_applied_when_not_class_agnostic(self):
        preds, refs = payload_to_det_metric(
            _make_payload(), label_mapping={"BOAT": 4}, class_agnostic=False
        )
        np.testing.assert_array_equal(preds[0]["labels"], [4])
        np.testing.assert_array_equal(refs[0]["labels"], [4])


# ---------------------------------------------------------------------------
# box_denormalize
# ---------------------------------------------------------------------------


class TestBoxDenormalize:
    """Relative -> pixel scaling for (N, 4) box arrays."""

    def test_empty_array_returned_unchanged(self):
        boxes = np.empty((0, 4))
        assert box_denormalize(boxes, 640, 480).size == 0

    def test_normalized_boxes_scaled_per_axis(self):
        boxes = np.array([[0.1, 0.2, 0.3, 0.4]])
        result = box_denormalize(boxes, 100, 50)
        np.testing.assert_allclose(result, [[10.0, 10.0, 30.0, 20.0]])

    def test_pixel_boxes_returned_unchanged(self):
        """Any value > 1.0 means the boxes are already in pixel space."""
        boxes = np.array([[10.0, 20.0, 30.0, 40.0]])
        np.testing.assert_array_equal(box_denormalize(boxes, 640, 480), boxes)

    def test_boundary_value_one_treated_as_normalized(self):
        boxes = np.array([[0.0, 0.0, 1.0, 1.0]])
        np.testing.assert_allclose(
            box_denormalize(boxes, 640, 512), [[0.0, 0.0, 640.0, 512.0]]
        )

    def test_input_array_is_not_mutated(self):
        """Regression: the scaling used to happen in place on the caller's array."""
        boxes = np.array([[0.1, 0.2, 0.3, 0.4]])
        original = boxes.copy()
        box_denormalize(boxes, 100, 50)
        np.testing.assert_array_equal(boxes, original)

    def test_integer_input_yields_float_output(self):
        """Regression: integer arrays must be cast before scaling."""
        boxes = np.array([[0, 0, 1, 1]])
        result = box_denormalize(boxes, 640, 512)
        assert result.dtype.kind == "f"
        np.testing.assert_allclose(result, [[0.0, 0.0, 640.0, 512.0]])

    def test_multiple_boxes_scaled_independently(self):
        boxes = np.array([[0.0, 0.0, 0.5, 0.5], [0.5, 0.25, 0.5, 0.75]])
        result = box_denormalize(boxes, 200, 100)
        np.testing.assert_allclose(
            result, [[0.0, 0.0, 100.0, 50.0], [100.0, 25.0, 100.0, 75.0]]
        )

    def test_all_zero_boxes_stay_zero(self):
        boxes = np.zeros((2, 4))
        np.testing.assert_allclose(box_denormalize(boxes, 640, 512), np.zeros((2, 4)))
