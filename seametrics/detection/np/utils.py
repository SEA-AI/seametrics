"""Numpy box conversion and input validation for the detection metrics."""

from typing import Dict, List, Optional, Sequence

import numpy as np

_BOX_COORDS = 4


def box_convert(boxes: np.ndarray, in_fmt: str, out_fmt: str) -> np.ndarray:
    """Converts boxes from given in_fmt to out_fmt.

    'xyxy': top-left corner x1, y1 and bottom-right corner x2, y2.
    'xywh': top-left corner x1, y1 and width, height.
    'cxcywh': center cx, cy and width, height.

    Args:
        boxes (np.ndarray[N, 4]): boxes which will be converted.
        in_fmt (str): Input format, one of 'xyxy', 'xywh', 'cxcywh'.
        out_fmt (str): Output format, one of 'xyxy', 'xywh', 'cxcywh'.

    Returns:
        np.ndarray[N, 4]: Converted boxes.
    """
    if boxes.size == 0:
        return boxes

    allowed_fmts = ("xyxy", "xywh", "cxcywh")
    if in_fmt not in allowed_fmts or out_fmt not in allowed_fmts:
        raise ValueError(
            "Unsupported Bounding Box Conversions for given in_fmt and out_fmt"
        )

    if in_fmt == out_fmt:
        return boxes.copy()

    if boxes.shape[-1] != _BOX_COORDS:
        raise ValueError(f"Expected boxes with 4 values each, got shape {boxes.shape}")

    # every conversion goes through xyxy
    if in_fmt != "xyxy":
        boxes = _TO_XYXY[in_fmt](boxes)
    if out_fmt != "xyxy":
        boxes = _FROM_XYXY[out_fmt](boxes)
    return boxes


def _box_xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert boxes from xywh to xyxy; formats as in box_convert."""
    x, y, w, h = boxes[..., 0:1], boxes[..., 1:2], boxes[..., 2:3], boxes[..., 3:4]
    return np.concatenate([x, y, x + w, y + h], axis=-1)


def _box_cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert boxes from cxcywh to xyxy; formats as in box_convert."""
    cx, cy, w, h = boxes[..., 0:1], boxes[..., 1:2], boxes[..., 2:3], boxes[..., 3:4]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return np.concatenate([x1, y1, x2, y2], axis=-1)


def _box_xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """Convert boxes from xyxy to xywh; formats as in box_convert."""
    x1, y1, x2, y2 = boxes[..., 0:1], boxes[..., 1:2], boxes[..., 2:3], boxes[..., 3:4]
    return np.concatenate([x1, y1, x2 - x1, y2 - y1], axis=-1)


def _box_xyxy_to_cxcywh(boxes: np.ndarray) -> np.ndarray:
    """Convert boxes from xyxy to cxcywh; formats as in box_convert."""
    x1, y1, x2, y2 = boxes[..., 0:1], boxes[..., 1:2], boxes[..., 2:3], boxes[..., 3:4]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return np.concatenate([cx, cy, w, h], axis=-1)


_TO_XYXY = {"xywh": _box_xywh_to_xyxy, "cxcywh": _box_cxcywh_to_xyxy}
_FROM_XYXY = {"xywh": _box_xyxy_to_xywh, "cxcywh": _box_xyxy_to_cxcywh}


def fix_empty_arrays(boxes: np.ndarray) -> np.ndarray:
    """Empty tensors can cause problems, this methods corrects them."""
    if boxes.size == 0 and boxes.ndim == 1:
        return np.expand_dims(boxes, axis=0)
    return boxes


def input_validator(
    preds: Sequence[Dict[str, np.ndarray]],
    targets: Sequence[Dict[str, np.ndarray]],
    iou_type: str = "bbox",
    labels: Optional[List[int]] = None,
) -> None:
    """Ensure the correct input format of `preds` and `targets`.

    Raises:
        ValueError: If ``iou_type`` is unsupported or the inputs are malformed.
    """
    if iou_type == "bbox":
        item_val_name = "boxes"
    elif iou_type == "segm":
        item_val_name = "masks"
    else:
        raise ValueError(f"IOU type {iou_type} is not supported")

    _check_containers(preds, targets)
    pred_keys = [item_val_name, "scores", "labels"]
    target_keys = [item_val_name, "labels"]
    _check_keys(preds, pred_keys, "preds")
    _check_keys(targets, target_keys, "targets")
    _check_types(preds, pred_keys, "preds")
    _check_types(targets, target_keys, "targets")

    for i, item in enumerate(targets):
        if item[item_val_name].shape[0] != item["labels"].shape[0]:
            raise ValueError(
                f"Input {item_val_name} and labels of sample {i} in targets have a"
                f" different length (expected {item[item_val_name].shape[0]} labels,"
                f" got {item['labels'].shape[0]})"
            )
        _check_known_labels(item, labels)
    for i, item in enumerate(preds):
        if not (
            item[item_val_name].shape[0]
            == item["labels"].shape[0]
            == item["scores"].shape[0]
        ):
            raise ValueError(
                f"Input {item_val_name}, labels and scores of sample {i} in"
                " predictions have a different length (expected"
                f" {item[item_val_name].shape[0]} labels and scores,"
                f" got {item['labels'].shape[0]} labels and"
                f" {item['scores'].shape[0]})"
            )
        _check_known_labels(item, labels)


def _check_containers(preds: object, targets: object) -> None:
    """Raise ValueError unless preds and targets are equal-length lists or tuples."""
    if not isinstance(preds, (list, tuple)):
        raise ValueError(
            "Expected argument `preds` to be of type list or tuple,"
            f" but got {type(preds)}"
        )
    if not isinstance(targets, (list, tuple)):
        raise ValueError(
            "Expected argument `targets` to be of type list or tuple,"
            f" but got {type(targets)}"
        )
    if len(preds) != len(targets):
        raise ValueError(
            "Expected argument `preds` and `targets` to have the same length,"
            f" but got {len(preds)} and {len(targets)}"
        )


def _check_keys(
    items: Sequence[Dict[str, np.ndarray]], keys: List[str], name: str
) -> None:
    """Raise ValueError unless every item holds every key."""
    for k in keys:
        if any(k not in item for item in items):
            raise ValueError(f"Expected all dicts in `{name}` to contain the `{k}` key")


def _check_types(
    items: Sequence[Dict[str, np.ndarray]], keys: List[str], name: str
) -> None:
    """Raise ValueError unless every value under ``keys`` is an ndarray."""
    for k in keys:
        if any(not isinstance(item[k], np.ndarray) for item in items):
            raise ValueError(f"Expected all {k} in `{name}` to be of type ndarray")


def _check_known_labels(
    item: Dict[str, np.ndarray], labels: Optional[List[int]]
) -> None:
    """Raise ValueError if ``labels`` is set and the item has a label outside it."""
    if labels and not set(item["labels"]).issubset(set(labels)):
        raise ValueError(
            f"Labels are predefined to be {labels},"
            f" but you provide unknown labels in {item['labels']}."
        )
