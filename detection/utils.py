import os
import contextlib
import io
from tqdm import tqdm
import numpy as np
import pandas as pd
import fiftyone as fo
from fiftyone import ViewField as F

from .imports import _TORCHMETRICS_AVAILABLE
if _TORCHMETRICS_AVAILABLE:
    from torch import tensor

# helper functions
# TODO: rethink placement of these functions, for now they are here


def prepare_data_for_det_metrics(gt_bboxes_per_frame,
                                 gt_labels_per_frame,
                                 dt_bboxes_per_frame,
                                 dt_labels_per_frame,
                                 dt_scores_per_frame,
                                 img_w: int = 640,
                                 img_h: int = 512):
    """
    Returns
    -------
    target, preds: tuple of list of dicts
        Each dict has keys "boxes", "labels", "scores" (scores is only in preds)
    """

    def _to_np_format(gt_bboxes_per_frame, gt_labels_per_frame,
                      dt_bboxes_per_frame, dt_labels_per_frame, dt_scores_per_frame,
                      img_w: int = 640, img_h: int = 512):
        """Converts a list of frames with detections (bboxes) to numpy format."""

        # put to numpy format
        target = []
        for boxes, labels in zip(gt_bboxes_per_frame,
                                 gt_labels_per_frame):
            target.append({
                "boxes": box_convert(
                    box_denormalize(boxes, img_w, img_h),
                    in_fmt="xywh",
                    out_fmt="xyxy"
                ),
                "labels": np.unique(labels, return_inverse=True)[1],
            })

        preds = []
        for boxes, labels, scores in zip(dt_bboxes_per_frame,
                                         dt_labels_per_frame,
                                         dt_scores_per_frame):
            preds.append({
                "boxes": box_convert(
                    box_denormalize(boxes, img_w, img_h),
                    in_fmt="xywh",
                    out_fmt="xyxy"
                ),
                "labels": np.unique(labels, return_inverse=True)[1],
                "scores": np.array(scores),
            })

        return target, preds

    def _to_tm_format(target, preds):
        for elem in target:  # frame-level
            for key, val in elem.items():
                if type(val) is np.ndarray:
                    elem[key] = tensor(val)
        for elem in preds:  # frame-level
            for key, val in elem.items():
                if type(val) is np.ndarray:
                    elem[key] = tensor(val)
        return target, preds

    def _validate_arrays(data, data_type: str):
        if data is None or len(data) == 0:
            data = [data]
        if data_type in ["bbox", "mask"]:
            if any([(_not_falsy(item) and not isinstance(item[0], (tuple, list, np.ndarray)))
                    for item in data]):
                data = [data]
        elif data_type in ["score", "label"]:
            if any([(_not_falsy(item) and not isinstance(item, (tuple, list, np.ndarray)))
                    for item in data]):
                data = [data]
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
        data = [np.array(x) if x is not None else np.array([])
                for x in data]
        return data

    def _not_falsy(x):
        if x is None:
            return False
        if isinstance(x, (list, tuple)) and len(x) == 0:
            return False
        if isinstance(x, np.ndarray) and x.size == 0:
            return False
        return True

    gt_bboxes_per_frame = _validate_arrays(
        gt_bboxes_per_frame, data_type="bbox")
    gt_labels_per_frame = _validate_arrays(
        gt_labels_per_frame, data_type="label")
    dt_bboxes_per_frame = _validate_arrays(
        dt_bboxes_per_frame, data_type="bbox")
    dt_labels_per_frame = _validate_arrays(
        dt_labels_per_frame, data_type="label")
    dt_scores_per_frame = _validate_arrays(
        dt_scores_per_frame, data_type="score")

    assert len(gt_bboxes_per_frame) == len(
        dt_bboxes_per_frame), "Number of frames in GT and prediction do not match" + \
        f" ({len(gt_bboxes_per_frame)} vs {len(dt_bboxes_per_frame)})"

    if all([item is None for sublist in dt_scores_per_frame for item in sublist]):
        # print("All scores are None, setting them to 1.0")
        dt_scores_per_frame = [[1.0] * len(x) for x in dt_scores_per_frame]

    target, preds = _to_np_format(
        gt_bboxes_per_frame, gt_labels_per_frame,
        dt_bboxes_per_frame, dt_labels_per_frame, dt_scores_per_frame)

    if _TORCHMETRICS_AVAILABLE:
        target, preds = _to_tm_format(target, preds)

    return target, preds


def get_relevant_fields(view: fo.DatasetView,
                        fields: list,  # fiftyone field names
                        ):
    """
    Returns a view with only the relevant fields to prevent memory issues.

    Parameters
    ----------
    view: fo.DatasetView
        Dataset view
    fields: list
        List of fiftyone field names. You can use dot notation (embedded.field.name).

    Returns
    -------
    fo.DatasetView
        Dataset view with only the relevant fields.
    """
    if view.media_type == 'video':
        return view.select_fields([f"frames.{f}" if view.has_frame_field(f) else f
                                   for f in fields])
    elif view.media_type == 'image':
        return view.select_fields(fields)
    else:
        raise ValueError(f"Unsupported media type: {view.media_type}")


def get_values(view: fo.DatasetView,
               field_name: str,  # fiftyone field name
               ):
    """
    Parameters
    ----------
    view: fo.DatasetView
        Dataset view
    field_name: str
        Fiftyone field name. You can use dot notation (embedded.field.name).

    Returns
    -------
    list
        List of values.
    """

    if view.media_type == 'video':
        return view.values(f"frames[].{field_name}")
    elif view.media_type == 'image':
        return view.values(field_name)
    else:
        raise ValueError(f"Unsupported media type: {view.media_type}")


def smart_compute_metrics(view: fo.DatasetView,
                          metric_fn: callable,   # torchmetrics metric
                          metric_kwargs: dict,   # kwargs for metric_fn
                          gt_field: str,         # fiftyone field name
                          pred_field: str,       # fiftyone field name
                          conf_thr: float = 0):  # confidence threshold
    """If the dataset is a video dataset, it updates the metric for each
    sequence and compute is called only once in the end. If the dataset is an
    image dataset, it computes the metric in a single pass."""

    # init metric
    metric = metric_fn(**metric_kwargs)
    print("Collecting bboxes, labels and scores...")

    if view.media_type == 'video':
        sequence_names = set(view.values("sequence"))
        for sequence_name in tqdm(sequence_names):
            target, preds = get_target_and_preds(
                (
                    view
                    .match(F("sequence") == sequence_name)
                ),
                gt_field,
                pred_field
            )
            metric.update(preds, target)

    elif view.media_type == 'image':
        target, preds = get_target_and_preds(view, gt_field, pred_field)
        metric.update(preds, target)

    else:
        raise ValueError(f"Unsupported media type: {view.media_type}")

    print("Computing metrics...")
    return metric.compute()


def get_target_and_preds(view: fo.DatasetView,
                         gt_field: str,  # fiftyone field name
                         pred_field: str,  # fiftyone field name
                         ):
    view = get_relevant_fields(view, [gt_field, pred_field])

    gt_bboxes_per_frame = get_values(view,
                                     f"{gt_field}.detections.bounding_box")
    gt_labels_per_frame = get_values(view,
                                     f"{gt_field}.detections.label")
    dt_bboxes_per_frame = get_values(view,
                                     f"{pred_field}.detections.bounding_box")
    dt_labels_per_frame = get_values(view,
                                     f"{pred_field}.detections.label")
    dt_scores_per_frame = get_values(view,
                                     f"{pred_field}.detections.confidence")

    target, preds = prepare_data_for_det_metrics(
        gt_bboxes_per_frame, gt_labels_per_frame,
        dt_bboxes_per_frame, dt_labels_per_frame, dt_scores_per_frame)

    return target, preds


def compute_metrics(view: fo.DatasetView,
                    gt_field: str,  # fiftyone field name
                    pred_field: str,  # fiftyone field name
                    metric_fn: callable,  # torchmetrics metric
                    metric_kwargs: dict):  # kwargs for metric_fn
    """Computes metrics for a given dataset view."""

    view = get_relevant_fields(view, [gt_field, pred_field])

    print("Collecting bboxes, labels and scores...")
    gt_bboxes_per_frame = get_values(view,
                                     f"{gt_field}.detections.bounding_box")
    gt_labels_per_frame = get_values(view,
                                     f"{gt_field}.detections.label")
    dt_bboxes_per_frame = get_values(view,
                                     f"{pred_field}.detections.bounding_box")
    dt_labels_per_frame = get_values(view,
                                     f"{pred_field}.detections.label")
    dt_scores_per_frame = get_values(view,
                                     f"{pred_field}.detections.confidence")

    print("Converting to metric format...")
    target, preds = prepare_data_for_det_metrics(
        gt_bboxes_per_frame, gt_labels_per_frame,
        dt_bboxes_per_frame, dt_labels_per_frame, dt_scores_per_frame)

    # free memory
    del gt_bboxes_per_frame, gt_labels_per_frame
    del dt_bboxes_per_frame, dt_labels_per_frame, dt_scores_per_frame

    print("Computing metrics...")
    metric = metric_fn(**metric_kwargs)
    metric.update(preds, target)
    return metric.compute()


def results_to_df(results, fixed_columns: dict = {}):
    # save to pandas dataframe
    columns = ["iou_threshold", "max_dets", "tp", "fp", "fn", "duplicates",
               "precision", "recall", "f1", "support", "fpi", "n_imgs"]
    columns = list(fixed_columns.keys()) + columns
    df = pd.DataFrame(columns=columns)

    for area_range_lbl, metric in results['metrics'].items():
        # print(f"{area_range_lbl}: {metric}")
        df.loc[len(df)] = {
            **fixed_columns,
            "area_range_lbl": area_range_lbl,
            "area_range": metric["range"],
            "iou_threshold": float(metric["iouThr"]),
            "max_dets": metric["maxDets"],
            "tp": metric["tp"],
            "fp": metric["fp"],
            "fn": metric["fn"],
            "duplicates": metric["duplicates"],
            "precision": metric["precision"],
            "recall": metric["recall"],
            "f1": metric["f1"],
            "support": metric["support"],
            "fpi": metric["fpi"],
            "n_imgs": metric["nImgs"],
        }

    return df


def sequence_results_to_df(sequence_results):
    # save to pandas dataframe
    columns = ["sequence", "area_range_lbl", "area_range", "iou_threshold", "max_dets",
               "tp", "fp", "fn", "duplicates", "precision", "recall", "f1", "support",
               "fpi", "n_imgs"]
    df = pd.DataFrame(columns=columns)

    for seq_name, results in sequence_results.items():
        for area_range_lbl, metric in results['metrics'].items():
            # print(f"{seq_name} - {area_range_lbl}: {metric}")
            df.loc[len(df)] = {
                "sequence": seq_name,
                "area_range_lbl": area_range_lbl,
                "area_range": metric["range"],
                "iou_threshold": float(metric["iouThr"]),
                "max_dets": metric["maxDets"],
                "tp": metric["tp"],
                "fp": metric["fp"],
                "fn": metric["fn"],
                "duplicates": metric["duplicates"],
                "precision": metric["precision"],
                "recall": metric["recall"],
                "f1": metric["f1"],
                "support": metric["support"],
                "fpi": metric["fpi"],
                "n_imgs": metric["nImgs"],
            }

    return df


def compute_and_save_sequence_metrics(
    csv_dirpath: str,
    view: fo.DatasetView,
    gt_field: str,  # fiftyone field name
    pred_field: str,  # fiftyone field name
    metric_fn: callable,  # torchmetrics metric
    metric_kwargs: dict,  # kwargs for metric_fn
    csv_suffix: str = None,
    debug: bool = False,
    name_separator: str = "__",
):
    csv_name = name_separator.join(
        [view.dataset_name, gt_field, pred_field, metric_fn.__name__])
    csv_name = name_separator.join(
        [csv_name, csv_suffix]) if csv_suffix else csv_name
    csv_name += ".csv"
    csv_path = os.path.join(csv_dirpath, csv_name)
    print(f"Saving metrics to {csv_path}")

    view = get_relevant_fields(view, [gt_field, pred_field, 'sequence'])

    sequence_results = {}
    sequence_names = view.distinct("sequence")
    for sequence_name in tqdm(sequence_names):

        with contextlib.redirect_stdout(io.StringIO()) as f:
            print(sequence_name)
            sequence_view = view.match(F("sequence") == sequence_name)
            sequence_results[sequence_name] = compute_metrics(
                view=sequence_view,
                gt_field=gt_field,
                pred_field=pred_field,
                metric_fn=metric_fn,
                metric_kwargs=metric_kwargs,
            )

        if debug:
            print(f.getvalue())

    # create csv dir if not exists
    if not os.path.exists(csv_dirpath):
        os.makedirs(csv_dirpath)
    df = sequence_results_to_df(sequence_results)
    df.to_csv(csv_path, index=False)


def get_confidence_metric_vals(cocoeval: np.ndarray,
                               T: int, R: int, K: int, A: int, M: int):
    """Get confidence values for plotting:
    - recall vs confidence
    - precision vs confidence
    - f1-score vs confidence

    Arguments
    ---------
    cocoeval: np.ndarray
        COCOeval object
    T: int
        iou threshold
    R: int
        recall threshold (not used so far)
    K: int
        catIds
    A: int
        area range index
    M: int
        max dets index

    Returns
    -------
    dict
        conf: confidence values
        p: precision values
        r: recall values
        f1: f1-score values
    """

    tpc = cocoeval['TPC'][T, K, A, M]
    fpc = cocoeval['FPC'][T, K, A, M]
    n_gt = cocoeval['TP'][T, K, A, M] + cocoeval['FN'][T, K, A, M]
    conf = cocoeval['sorted_conf'][K, A, M]
    eps = 1e-16
    x = np.linspace(0, 1, 1000)  # for plotting

    # Recall
    recall = tpc / (n_gt + eps)  # recall curve
    # negative x, xp because xp decreases
    r = np.interp(-x, -conf, recall, left=0)

    # Precision
    precision = tpc / (tpc + fpc)  # precision curve
    p = np.interp(-x, -conf, precision, left=1)  # p at pr_score

    # F1-score
    f1 = 2 * p * r / (p + r + eps)

    return dict(conf=x, precision=p, recall=r, f1=f1)


def box_denormalize(boxes: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """
    Denormalizes boxes from [0, 1] to [0, img_w] and [0, img_h].
    Args:
        boxes (Tensor[N, 4]): boxes which will be denormalized.
        img_w (int): Width of image.
        img_h (int): Height of image.

    Returns:
        Tensor[N, 4]: Denormalized boxes.
    """
    if boxes.size == 0:
        return boxes

    # check if boxes are normalized
    if np.any(boxes > 1.0):
        return boxes

    boxes[:, 0::2] *= img_w
    boxes[:, 1::2] *= img_h
    return boxes


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
            "Unsupported Bounding Box Conversions for given in_fmt and out_fmt")

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
            f"Expected argument `preds` to be of type list or tuple, but got {type(preds)}")
    if not isinstance(targets, (list, tuple)):
        raise ValueError(
            f"Expected argument `targets` to be of type list or tuple, but got {type(targets)}")
    if len(preds) != len(targets):
        raise ValueError(
            f"Expected argument `preds` and `targets` to have the same length, but got {len(preds)} and {len(targets)}"
        )

    for k in [item_val_name, "scores", "labels"]:
        if any(k not in p for p in preds):
            raise ValueError(
                f"Expected all dicts in `preds` to contain the `{k}` key")

    for k in [item_val_name, "labels"]:
        if any(k not in p for p in targets):
            raise ValueError(
                f"Expected all dicts in `targets` to contain the `{k}` key")

    if any(type(pred[item_val_name]) is not np.ndarray for pred in preds):
        raise ValueError(
            f"Expected all {item_val_name} in `preds` to be of type ndarray")
    if any(type(pred["scores"]) is not np.ndarray for pred in preds):
        raise ValueError(
            "Expected all scores in `preds` to be of type ndarray")
    if any(type(pred["labels"]) is not np.ndarray for pred in preds):
        raise ValueError(
            "Expected all labels in `preds` to be of type ndarray")
    if any(type(target[item_val_name]) is not np.ndarray for target in targets):
        raise ValueError(
            f"Expected all {item_val_name} in `targets` to be of type ndarray")
    if any(type(target["labels"]) is not np.ndarray for target in targets):
        raise ValueError(
            "Expected all labels in `targets` to be of type ndarray")

    for i, item in enumerate(targets):
        if item[item_val_name].shape[0] != item["labels"].shape[0]:
            raise ValueError(
                f"Input {item_val_name} and labels of sample {i} in targets have a"
                f" different length (expected {item[item_val_name].shape[0]} labels, got {item['labels'].shape[0]})"
            )
    for i, item in enumerate(preds):
        if not (item[item_val_name].shape[0] == item["labels"].shape[0] == item["scores"].shape[0]):
            raise ValueError(
                f"Input {item_val_name}, labels and scores of sample {i} in predictions have a"
                f" different length (expected {item[item_val_name].shape[0]} labels and scores,"
                f" got {item['labels'].shape[0]} labels and {item['scores'].shape[0]})"
            )
