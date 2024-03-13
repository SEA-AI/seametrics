import numpy as np


def box_convert(boxes: np.ndarray, in_fmt: str, out_fmt: str) -> np.ndarray:
    """
    Converts boxes from given in_fmt to out_fmt.
    Supported in_fmt and out_fmt are:

    'xyxy': boxes are represented via corners, x1, y1 being top left and x2, y2 being bottom right.
    This is the format that torchvision utilities expect.

    'xywh' : boxes are represented via corner, width and height, x1, y2 being top left, w, h being width and height.

    'cxcywh' : boxes are represented via centre, width and height, cx, cy being center of box, w, h
    being width and height.

    Args:
        boxes (Tensor[N, 4]): boxes which will be converted.
        in_fmt (str): Input format of given boxes. Supported formats are ['xyxy', 'xywh', 'cxcywh'].
        out_fmt (str): Output format of given boxes. Supported formats are ['xyxy', 'xywh', 'cxcywh']

    Returns:
        Tensor[N, 4]: Boxes into converted format.
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
        # convert to xyxy and change in_fmt xyxy
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


def _box_xywh_to_xyxy(boxes):
    """
    Converts bounding boxes from (x, y, w, h) format to (x1, y1, x2, y2) format.
    (x, y) refers to top left of bounding box.
    (w, h) refers to width and height of box.
    Args:
        boxes (ndarray[N, 4]): boxes in (x, y, w, h) which will be converted.

    Returns:
        boxes (ndarray[N, 4]): boxes in (x1, y1, x2, y2) format.
    """
    x, y, w, h = np.split(boxes, 4, axis=-1)
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    converted_boxes = np.concatenate([x1, y1, x2, y2], axis=-1)
    return converted_boxes


def _box_cxcywh_to_xyxy(boxes):
    """
    Converts bounding boxes from (cx, cy, w, h) format to (x1, y1, x2, y2) format.
    (cx, cy) refers to center of bounding box
    (w, h) are width and height of bounding box
    Args:
        boxes (ndarray[N, 4]): boxes in (cx, cy, w, h) format which will be converted.

    Returns:
        boxes (ndarray[N, 4]): boxes in (x1, y1, x2, y2) format.
    """
    cx, cy, w, h = np.split(boxes, 4, axis=-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    converted_boxes = np.concatenate([x1, y1, x2, y2], axis=-1)
    return converted_boxes


def _box_xyxy_to_xywh(boxes):
    """
    Converts bounding boxes from (x1, y1, x2, y2) format to (x, y, w, h) format.
    (x1, y1) refer to top left of bounding box
    (x2, y2) refer to bottom right of bounding box
    Args:
        boxes (ndarray[N, 4]): boxes in (x1, y1, x2, y2) which will be converted.

    Returns:
        boxes (ndarray[N, 4]): boxes in (x, y, w, h) format.
    """
    x1, y1, x2, y2 = np.split(boxes, 4, axis=-1)
    w = x2 - x1
    h = y2 - y1
    converted_boxes = np.concatenate([x1, y1, w, h], axis=-1)
    return converted_boxes


def _box_xyxy_to_cxcywh(boxes):
    """
    Converts bounding boxes from (x1, y1, x2, y2) format to (cx, cy, w, h) format.
    (x1, y1) refer to top left of bounding box
    (x2, y2) refer to bottom right of bounding box
    Args:
        boxes (ndarray[N, 4]): boxes in (x1, y1, x2, y2) format which will be converted.

    Returns:
        boxes (ndarray[N, 4]): boxes in (cx, cy, w, h) format.
    """
    x1, y1, x2, y2 = np.split(boxes, 4, axis=-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    converted_boxes = np.concatenate([cx, cy, w, h], axis=-1)
    return converted_boxes


def _fix_empty_arrays(boxes: np.ndarray) -> np.ndarray:
    """Empty tensors can cause problems, this methods corrects them."""
    if boxes.size == 0 and boxes.ndim == 1:
        return np.expand_dims(boxes, axis=0)
    return boxes


def _input_validator(preds, targets, iou_type="bbox"):
    """Ensure the correct input format of `preds` and `targets`."""
    if iou_type == "bbox":
        item_val_name = "boxes"
    elif iou_type == "segm":
        item_val_name = "masks"
    else:
        raise Exception(f"IOU type {iou_type} is not supported")

    if not isinstance(preds, (list, tuple)):
        raise ValueError(
            f"Expected argument `preds` to be of type list or tuple, but got {type(preds)}"
        )
    if not isinstance(targets, (list, tuple)):
        raise ValueError(
            f"Expected argument `targets` to be of type list or tuple, but got {type(targets)}"
        )
    if len(preds) != len(targets):
        raise ValueError(
            f"Expected argument `preds` and `targets` to have the same length, but got {len(preds)} and {len(targets)}"
        )

    for k in [item_val_name, "scores", "labels"]:
        if any(k not in p for p in preds):
            raise ValueError(f"Expected all dicts in `preds` to contain the `{k}` key")

    for k in [item_val_name, "labels"]:
        if any(k not in p for p in targets):
            raise ValueError(
                f"Expected all dicts in `targets` to contain the `{k}` key"
            )

    if any(type(pred[item_val_name]) is not np.ndarray for pred in preds):
        raise ValueError(
            f"Expected all {item_val_name} in `preds` to be of type ndarray"
        )
    if any(type(pred["scores"]) is not np.ndarray for pred in preds):
        raise ValueError("Expected all scores in `preds` to be of type ndarray")
    if any(type(pred["labels"]) is not np.ndarray for pred in preds):
        raise ValueError("Expected all labels in `preds` to be of type ndarray")
    if any(type(target[item_val_name]) is not np.ndarray for target in targets):
        raise ValueError(
            f"Expected all {item_val_name} in `targets` to be of type ndarray"
        )
    if any(type(target["labels"]) is not np.ndarray for target in targets):
        raise ValueError("Expected all labels in `targets` to be of type ndarray")

    for i, item in enumerate(targets):
        if item[item_val_name].shape[0] != item["labels"].shape[0]:
            raise ValueError(
                f"Input {item_val_name} and labels of sample {i} in targets have a"
                f" different length (expected {item[item_val_name].shape[0]} labels, got {item['labels'].shape[0]})"
            )
    for i, item in enumerate(preds):
        if not (
            item[item_val_name].shape[0]
            == item["labels"].shape[0]
            == item["scores"].shape[0]
        ):
            raise ValueError(
                f"Input {item_val_name}, labels and scores of sample {i} in predictions have a"
                f" different length (expected {item[item_val_name].shape[0]} labels and scores,"
                f" got {item['labels'].shape[0]} labels and {item['scores'].shape[0]})"
            )
