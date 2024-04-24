import os
import contextlib
import io
from typing import Dict, List, Tuple
from tqdm import tqdm
import numpy as np
import pandas as pd
import fiftyone as fo
from fiftyone import ViewField as F

from seametrics.payload import Payload
from seametrics.detection.np.utils import box_convert
from seametrics.detection.imports import _TORCHMETRICS_AVAILABLE

if _TORCHMETRICS_AVAILABLE:
    from torch import tensor


# payload functions


def payload_to_det_metric(
    payload: Payload,
    model_name: str = None,
) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]]]:
    """
    Convert the payload data to detection metrics format.

    Args:
        payload (Dict): The payload data containing sequences, models,
            and ground truth field name.
        model_name (str, optional): The name of the model. If not provided,
            the first model in the payload will be used.

    Returns:
        Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]]]:
            A tuple containing the converted (predictions, references).
    """
    predictions, references = [], []

    if model_name is None:
        model_name = payload.models[0]

    for _, sequence in payload.sequences.items():
        w, h = (
            sequence.resolution.width,
            sequence.resolution.height,
        )
        predictions.extend(payload_sequence_to_det_metrics(sequence[model_name], w, h))
        references.extend(
            payload_sequence_to_det_metrics(
                sequence[payload.gt_field_name], w, h, is_gt=True
            )
        )

    return predictions, references


def payload_sequence_to_det_metrics(
    sequence_dets: List[List[fo.Detection]],
    w: int,
    h: int,
    is_gt: bool = False,
) -> List[Dict[str, np.ndarray]]:
    """
    Convert a sequence of detections to the format required by the
    PrecisionRecallF1Support() function of the seametrics library.

    Args:
        sequence_dets (List[List[fo.Detection]]): A list of fiftyone detections.
        w (int): Width in pixels of the image.
        h (int): Height in pixels of the image.
        is_gt (bool, optional): Flag indicating if the input data is ground truth.
            Defaults to False.

    Returns:
        List[Dict[str, np.ndarray]]: A list containing the converted detections.
    """
    output = []

    for frame_dets in sequence_dets:
        frame_dict = frame_dets_to_det_metrics(frame_dets, w, h, is_gt)
        output.append(frame_dict)

    return output


def frame_dets_to_det_metrics(
    fo_dets: List[fo.Detection],
    w: int,
    h: int,
    is_gt: bool = False,
    class_agnostic: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Convert a list of fiftyone detections to the format required by the
    PrecisionRecallF1Support() function of the seametrics library.

    Args:
        fo_dets (List[fo.Detection]): A list of fiftyone detections.
        w (int): Width in pixels of the image.
        h (int): Height in pixels of the image.
        is_gt (bool, optional): Flag indicating if the input data is ground truth.
            Defaults to False.
        class_agnostic (bool, optional): Flag indicating if class agnostic mode is
            enabled. Defaults to True.

    Returns:
        Dict[str, np.ndarray]: A dictionary containing the converted detections.
    """
    if not class_agnostic:
        raise ValueError("Only class agnostic mode is supported")

    detections = []
    labels = []
    scores = []
    areas = []

    for det in fo_dets:
        bbox = det["bounding_box"]

        detections.append([bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h])
        labels.append(0 if class_agnostic else det["label"])
        scores.append(det["confidence"] if det["confidence"] else 1.0)  # None for gt

        if is_gt:
            if "area" not in det.field_names:
                msg = (
                    "Area not found in ground truth detections. "
                    "Please make sure that area is included in ground truth detections."
                )
                raise ValueError(msg)
            areas.append(det["area"] if "area" in det.field_names else -1)

    metrics_dict = {
        "boxes": np.array(detections),
        "labels": np.array(labels),
    }
    if is_gt:
        metrics_dict["area"] = np.array(areas)
    else:
        metrics_dict["scores"] = np.array(scores)
    return metrics_dict


# DEPRECATED helper functions


def prepare_data_for_det_metrics(
    gt_bboxes_per_frame,
    gt_labels_per_frame,
    dt_bboxes_per_frame,
    dt_labels_per_frame,
    dt_scores_per_frame,
    img_w: int = 640,
    img_h: int = 512,
):
    """
    Returns
    -------
    target, preds: tuple of list of dicts
        Each dict has keys "boxes", "labels", "scores" (scores is only in preds)
    """

    def _to_np_format(
        gt_bboxes_per_frame,
        gt_labels_per_frame,
        dt_bboxes_per_frame,
        dt_labels_per_frame,
        dt_scores_per_frame,
        img_w: int = 640,
        img_h: int = 512,
    ):
        """Converts a list of frames with detections (bboxes) to numpy format."""

        # put to numpy format
        target = []
        for boxes, labels in zip(gt_bboxes_per_frame, gt_labels_per_frame):
            target.append(
                {
                    "boxes": box_convert(
                        box_denormalize(boxes, img_w, img_h),
                        in_fmt="xywh",
                        out_fmt="xyxy",
                    ),
                    "labels": np.unique(labels, return_inverse=True)[1],
                }
            )

        preds = []
        for boxes, labels, scores in zip(
            dt_bboxes_per_frame, dt_labels_per_frame, dt_scores_per_frame
        ):
            preds.append(
                {
                    "boxes": box_convert(
                        box_denormalize(boxes, img_w, img_h),
                        in_fmt="xywh",
                        out_fmt="xyxy",
                    ),
                    "labels": np.unique(labels, return_inverse=True)[1],
                    "scores": np.array(scores),
                }
            )

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
            if any(
                [
                    (
                        _not_falsy(item)
                        and not isinstance(item[0], (tuple, list, np.ndarray))
                    )
                    for item in data
                ]
            ):
                data = [data]
        elif data_type in ["score", "label"]:
            if any(
                [
                    (
                        _not_falsy(item)
                        and not isinstance(item, (tuple, list, np.ndarray))
                    )
                    for item in data
                ]
            ):
                data = [data]
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
        data = [np.array(x) if x is not None else np.array([]) for x in data]
        return data

    def _not_falsy(x):
        if x is None:
            return False
        if isinstance(x, (list, tuple)) and len(x) == 0:
            return False
        if isinstance(x, np.ndarray) and x.size == 0:
            return False
        return True

    gt_bboxes_per_frame = _validate_arrays(gt_bboxes_per_frame, data_type="bbox")
    gt_labels_per_frame = _validate_arrays(gt_labels_per_frame, data_type="label")
    dt_bboxes_per_frame = _validate_arrays(dt_bboxes_per_frame, data_type="bbox")
    dt_labels_per_frame = _validate_arrays(dt_labels_per_frame, data_type="label")
    dt_scores_per_frame = _validate_arrays(dt_scores_per_frame, data_type="score")

    assert len(gt_bboxes_per_frame) == len(dt_bboxes_per_frame), (
        "Number of frames in GT and prediction do not match"
        + f" ({len(gt_bboxes_per_frame)} vs {len(dt_bboxes_per_frame)})"
    )

    if all([item is None for sublist in dt_scores_per_frame for item in sublist]):
        # print("All scores are None, setting them to 1.0")
        dt_scores_per_frame = [[1.0] * len(x) for x in dt_scores_per_frame]

    target, preds = _to_np_format(
        gt_bboxes_per_frame,
        gt_labels_per_frame,
        dt_bboxes_per_frame,
        dt_labels_per_frame,
        dt_scores_per_frame,
    )

    if _TORCHMETRICS_AVAILABLE:
        target, preds = _to_tm_format(target, preds)

    return target, preds


def get_relevant_fields(
    view: fo.DatasetView,
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
    if view.media_type == "video":
        return view.select_fields(
            [f"frames.{f}" if view.has_frame_field(f) else f for f in fields]
        )
    elif view.media_type == "image":
        return view.select_fields(fields)
    else:
        raise ValueError(f"Unsupported media type: {view.media_type}")


def get_values(
    view: fo.DatasetView,
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

    if view.media_type == "video":
        return view.values(f"frames[].{field_name}")
    elif view.media_type == "image":
        return view.values(field_name)
    else:
        raise ValueError(f"Unsupported media type: {view.media_type}")


def smart_compute_metrics(
    view: fo.DatasetView,
    metric_fn: callable,  # torchmetrics metric
    metric_kwargs: dict,  # kwargs for metric_fn
    gt_field: str,  # fiftyone field name
    pred_field: str,  # fiftyone field name
    conf_thr: float = 0,
):  # confidence threshold
    """If the dataset is a video dataset, it updates the metric for each
    sequence and compute is called only once in the end. If the dataset is an
    image dataset, it computes the metric in a single pass."""

    # init metric
    metric = metric_fn(**metric_kwargs)
    print("Collecting bboxes, labels and scores...")

    if view.media_type == "video":
        sequence_names = set(view.values("sequence"))
        for sequence_name in tqdm(sequence_names):
            target, preds = get_target_and_preds(
                (view.match(F("sequence") == sequence_name)), gt_field, pred_field
            )
            metric.update(preds, target)

    elif view.media_type == "image":
        target, preds = get_target_and_preds(view, gt_field, pred_field)
        metric.update(preds, target)

    else:
        raise ValueError(f"Unsupported media type: {view.media_type}")

    print("Computing metrics...")
    return metric.compute()


def get_target_and_preds(
    view: fo.DatasetView,
    gt_field: str,  # fiftyone field name
    pred_field: str,  # fiftyone field name
):
    view = get_relevant_fields(view, [gt_field, pred_field])

    gt_bboxes_per_frame = get_values(view, f"{gt_field}.detections.bounding_box")
    gt_labels_per_frame = get_values(view, f"{gt_field}.detections.label")
    dt_bboxes_per_frame = get_values(view, f"{pred_field}.detections.bounding_box")
    dt_labels_per_frame = get_values(view, f"{pred_field}.detections.label")
    dt_scores_per_frame = get_values(view, f"{pred_field}.detections.confidence")

    target, preds = prepare_data_for_det_metrics(
        gt_bboxes_per_frame,
        gt_labels_per_frame,
        dt_bboxes_per_frame,
        dt_labels_per_frame,
        dt_scores_per_frame,
    )

    return target, preds


def compute_metrics(
    view: fo.DatasetView,
    gt_field: str,  # fiftyone field name
    pred_field: str,  # fiftyone field name
    metric_fn: callable,  # torchmetrics metric
    metric_kwargs: dict,
):  # kwargs for metric_fn
    """Computes metrics for a given dataset view."""

    view = get_relevant_fields(view, [gt_field, pred_field])

    print("Collecting bboxes, labels and scores...")
    gt_bboxes_per_frame = get_values(view, f"{gt_field}.detections.bounding_box")
    gt_labels_per_frame = get_values(view, f"{gt_field}.detections.label")
    dt_bboxes_per_frame = get_values(view, f"{pred_field}.detections.bounding_box")
    dt_labels_per_frame = get_values(view, f"{pred_field}.detections.label")
    dt_scores_per_frame = get_values(view, f"{pred_field}.detections.confidence")

    print("Converting to metric format...")
    target, preds = prepare_data_for_det_metrics(
        gt_bboxes_per_frame,
        gt_labels_per_frame,
        dt_bboxes_per_frame,
        dt_labels_per_frame,
        dt_scores_per_frame,
    )

    # free memory
    del gt_bboxes_per_frame, gt_labels_per_frame
    del dt_bboxes_per_frame, dt_labels_per_frame, dt_scores_per_frame

    print("Computing metrics...")
    metric = metric_fn(**metric_kwargs)
    metric.update(preds, target)
    return metric.compute()


def results_to_df(results, fixed_columns: dict = {}):
    # save to pandas dataframe
    columns = [
        "iou_threshold",
        "max_dets",
        "tp",
        "fp",
        "fn",
        "duplicates",
        "precision",
        "recall",
        "f1",
        "support",
        "fpi",
        "n_imgs",
    ]
    columns = list(fixed_columns.keys()) + columns
    df = pd.DataFrame(columns=columns)

    for area_range_lbl, metric in results["metrics"].items():
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
    columns = [
        "sequence",
        "area_range_lbl",
        "area_range",
        "iou_threshold",
        "max_dets",
        "tp",
        "fp",
        "fn",
        "duplicates",
        "precision",
        "recall",
        "f1",
        "support",
        "fpi",
        "n_imgs",
    ]
    df = pd.DataFrame(columns=columns)

    for seq_name, results in sequence_results.items():
        for area_range_lbl, metric in results["metrics"].items():
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
        [view.dataset_name, gt_field, pred_field, metric_fn.__name__]
    )
    csv_name = name_separator.join([csv_name, csv_suffix]) if csv_suffix else csv_name
    csv_name += ".csv"
    csv_path = os.path.join(csv_dirpath, csv_name)
    print(f"Saving metrics to {csv_path}")

    view = get_relevant_fields(view, [gt_field, pred_field, "sequence"])

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


def get_confidence_metric_vals(
    cocoeval: np.ndarray, T: int, R: int, K: int, A: int, M: int
):
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

    tpc = cocoeval["TPC"][T, K, A, M]
    fpc = cocoeval["FPC"][T, K, A, M]
    n_gt = cocoeval["TP"][T, K, A, M] + cocoeval["FN"][T, K, A, M]
    conf = cocoeval["sorted_conf"][K, A, M]
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
