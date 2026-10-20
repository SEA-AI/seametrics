import numpy as np
import pytest

from seametrics.detection.np.utils import box_convert, input_validator

# one box, x1=10 y1=20 x2=40 y2=60 -> w=30 h=40, center (25, 40)
XYXY = np.array([[10.0, 20.0, 40.0, 60.0]])
XYWH = np.array([[10.0, 20.0, 30.0, 40.0]])
CXCYWH = np.array([[25.0, 40.0, 30.0, 40.0]])
BOXES = {"xyxy": XYXY, "xywh": XYWH, "cxcywh": CXCYWH}


@pytest.mark.parametrize("in_fmt", BOXES)
@pytest.mark.parametrize("out_fmt", BOXES)
def test_box_convert_every_pair(in_fmt, out_fmt):
    out = box_convert(BOXES[in_fmt], in_fmt=in_fmt, out_fmt=out_fmt)
    np.testing.assert_array_equal(out, BOXES[out_fmt])


def test_box_convert_same_format_returns_copy():
    out = box_convert(XYXY, in_fmt="xyxy", out_fmt="xyxy")
    assert out is not XYXY


def test_box_convert_rejects_unknown_format():
    with pytest.raises(ValueError, match="Unsupported Bounding Box Conversions"):
        box_convert(XYXY, in_fmt="xyxy", out_fmt="yxyx")


def _pred():
    return {"boxes": XYXY, "scores": np.array([0.5]), "labels": np.array([0])}


def _target():
    return {"boxes": XYXY, "labels": np.array([0])}


@pytest.mark.parametrize(
    ("preds", "targets", "match"),
    [
        ([_pred()], [], "same length"),
        ([{"boxes": XYXY, "labels": np.array([0])}], [_target()], "`scores` key"),
        ([_pred()], [{"boxes": XYXY}], "`targets` to contain the `labels`"),
        ([{**_pred(), "scores": [0.5]}], [_target()], "scores in `preds`"),
        ([_pred()], [{**_target(), "labels": np.array([0, 1])}], "sample 0 in targets"),
        ([{**_pred(), "scores": np.array([0.5, 0.1])}], [_target()], "predictions"),
    ],
)
def test_input_validator_rejects(preds, targets, match):
    with pytest.raises(ValueError, match=match):
        input_validator(preds, targets)


def test_input_validator_rejects_unknown_labels():
    with pytest.raises(ValueError, match="unknown labels"):
        input_validator([_pred()], [{**_target(), "labels": np.array([3])}], labels=[0])


def test_input_validator_accepts_valid_input():
    input_validator([_pred()], [_target()], labels=[0])


def test_input_validator_rejects_unknown_iou_type():
    with pytest.raises(ValueError, match="IOU type keypoints is not supported"):
        input_validator([_pred()], [_target()], iou_type="keypoints")


@pytest.mark.parametrize(
    ("preds", "targets", "match"),
    [
        (_pred(), [_target()], "`preds` to be of type list or tuple"),
        ([_pred()], _target(), "`targets` to be of type list or tuple"),
    ],
)
def test_input_validator_rejects_non_sequences(preds, targets, match):
    with pytest.raises(ValueError, match=match):
        input_validator(preds, targets)


def test_box_convert_rejects_boxes_without_four_values():
    with pytest.raises(ValueError, match="4 values each"):
        box_convert(
            np.array([[1.0, 2.0, 3.0, 4.0, 9.0]]), in_fmt="xyxy", out_fmt="xywh"
        )
