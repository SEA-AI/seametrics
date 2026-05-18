"""Pure-numpy bounding-box conversion and denormalisation helpers."""

from __future__ import annotations

import numpy as np


def box_denormalize(boxes: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Denormalize boxes from [0, 1] to pixel coordinates.

    Args:
        boxes: Array of boxes to denormalize (shape ``[N, 4]``).
        img_w: Image width in pixels.
        img_h: Image height in pixels.

    Returns:
        Array of denormalized boxes with x-coordinates scaled by *img_w* and
        y-coordinates scaled by *img_h*.
    """
    if boxes.size == 0:
        return boxes

    if np.any(boxes > 1.0):
        return boxes

    boxes[0::2] *= img_w
    boxes[1::2] *= img_h
    return boxes


def box_convert(boxes: np.ndarray, in_fmt: str, out_fmt: str) -> np.ndarray:  # noqa: C901
    """Convert boxes from one format to another.

    Supported formats:

    ``'xyxy'``: boxes are represented via corners, x1, y1 being top left and
    x2, y2 being bottom right. This is the format that torchvision utilities
    expect.

    ``'xywh'``: boxes are represented via corner, width and height, x1, y1
    being top left, w, h being width and height.

    ``'cxcywh'``: boxes are represented via centre, width and height, cx, cy
    being center of box, w, h being width and height.

    Args:
        boxes: Boxes which will be converted (shape ``[N, 4]``).
        in_fmt: Input format of given boxes. Supported formats are
            ``['xyxy', 'xywh', 'cxcywh']``.
        out_fmt: Output format of given boxes. Supported formats are
            ``['xyxy', 'xywh', 'cxcywh']``.

    Returns:
        Boxes converted to *out_fmt* (shape ``[N, 4]``).

    Raises:
        ValueError: If *in_fmt* or *out_fmt* is not a supported format string.
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

    if in_fmt != "xyxy" and out_fmt != "xyxy":
        if in_fmt == "xywh":
            boxes = _box_xywh_to_xyxy(boxes)
        elif in_fmt == "cxcywh":
            boxes = _box_cxcywh_to_xyxy(boxes)
        in_fmt = "xyxy"

    if in_fmt == "xyxy":
        if out_fmt == "xywh":
            boxes = _box_xyxy_to_xywh(boxes)
        elif out_fmt == "cxcywh":
            boxes = _box_xyxy_to_cxcywh(boxes)
    elif out_fmt == "xyxy":
        if in_fmt == "xywh":
            boxes = _box_xywh_to_xyxy(boxes)
        elif in_fmt == "cxcywh":
            boxes = _box_cxcywh_to_xyxy(boxes)
    return boxes


def _box_xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert bounding boxes from (x, y, w, h) format to (x1, y1, x2, y2) format.

    (x, y) refers to top left of bounding box.
    (w, h) refers to width and height of box.

    Args:
        boxes: Boxes in (x, y, w, h) format (shape ``[N, 4]``).

    Returns:
        Boxes in (x1, y1, x2, y2) format (shape ``[N, 4]``).
    """
    x = boxes[..., 0:1]
    y = boxes[..., 1:2]
    w = boxes[..., 2:3]
    h = boxes[..., 3:4]
    converted_boxes = np.concatenate([x, y, x + w, y + h], axis=-1)
    return converted_boxes


def _box_cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert bounding boxes from (cx, cy, w, h) format to (x1, y1, x2, y2) format.

    (cx, cy) refers to center of bounding box.
    (w, h) are width and height of bounding box.

    Args:
        boxes: Boxes in (cx, cy, w, h) format (shape ``[N, 4]``).

    Returns:
        Boxes in (x1, y1, x2, y2) format (shape ``[N, 4]``).
    """
    cx = boxes[..., 0:1]
    cy = boxes[..., 1:2]
    w = boxes[..., 2:3]
    h = boxes[..., 3:4]
    converted_boxes = np.concatenate(
        [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], axis=-1
    )
    return converted_boxes


def _box_xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """Convert bounding boxes from (x1, y1, x2, y2) format to (x, y, w, h) format.

    (x1, y1) refer to top left of bounding box.
    (x2, y2) refer to bottom right of bounding box.

    Args:
        boxes: Boxes in (x1, y1, x2, y2) format (shape ``[N, 4]``).

    Returns:
        Boxes in (x, y, w, h) format (shape ``[N, 4]``).
    """
    x1 = boxes[..., 0:1]
    y1 = boxes[..., 1:2]
    x2 = boxes[..., 2:3]
    y2 = boxes[..., 3:4]
    converted_boxes = np.concatenate([x1, y1, x2 - x1, y2 - y1], axis=-1)
    return converted_boxes


def _box_xyxy_to_cxcywh(boxes: np.ndarray) -> np.ndarray:
    """Convert bounding boxes from (x1, y1, x2, y2) format to (cx, cy, w, h) format.

    (x1, y1) refer to top left of bounding box.
    (x2, y2) refer to bottom right of bounding box.

    Args:
        boxes: Boxes in (x1, y1, x2, y2) format (shape ``[N, 4]``).

    Returns:
        Boxes in (cx, cy, w, h) format (shape ``[N, 4]``).
    """
    x1 = boxes[..., 0:1]
    y1 = boxes[..., 1:2]
    x2 = boxes[..., 2:3]
    y2 = boxes[..., 3:4]
    converted_boxes = np.concatenate(
        [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=-1
    )
    return converted_boxes
