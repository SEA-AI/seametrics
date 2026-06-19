import contextlib
import io
import os
import pathlib
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import fiftyone as fo
import numpy as np
import pandas as pd
from deprecated import deprecated
from fiftyone import ViewField as F
from tqdm import tqdm

from seametrics.detection.imports import _TORCHMETRICS_AVAILABLE
from seametrics.detection.np.utils import box_convert
from seametrics.payload import Payload

if TYPE_CHECKING:
    from seametrics.detection.det_metrics import DetectionMetrics

if _TORCHMETRICS_AVAILABLE:
    from torch import tensor

# payload functions

error_code = None


def payload_to_det_metric(
    payload: Payload,
    model_name: str = None,
    label_mapping: Dict[str, int] = None,
    class_agnostic: bool = True,
) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]]]:
    """Convert the payload data to detection metrics format.

    Args:
        payload (Dict): The payload data containing sequences, models,
            and ground truth field name.
        model_name (str, optional): The name of the model. If not provided,
            the first model in the payload will be used.
        label_mapping (Dict[str, int], optional): Dictionary mapping string labels to
            numbers, which should be provided if the detection metrics should be
            calculated in a class-specific way. Defaults to None.
        class_agnostic (bool, optional): Flag indicating if the metrics should be
            calculated in a class-agnostic way. Defaults to True.

    Returns:
        Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]]]:
            A tuple containing the converted (predictions, references).
    """
    if class_agnostic and label_mapping is not None:
        raise ValueError("Label mapping cannot be provided for class-agnostic metrics.")

    predictions, references = [], []

    if model_name is None:
        model_name = payload.models[0]

    for _, sequence in payload.sequences.items():
        w, h = (
            sequence.resolution.width,
            sequence.resolution.height,
        )
        predictions.extend(
            payload_sequence_to_det_metrics(
                sequence_dets=sequence[model_name],
                w=w,
                h=h,
                label_mapping=label_mapping
            )
        )
        references.extend(
            payload_sequence_to_det_metrics(
                sequence_dets=sequence[payload.gt_field_name],
                w=w,
                h=h,
                is_gt=True,
                label_mapping=label_mapping
            )
        )

    return predictions, references


def payload_sequence_to_det_metrics(
    sequence_dets: List[List[fo.Detection]],
    w: int,
    h: int,
    is_gt: bool = False,
    label_mapping: Dict[str, int] = None,
) -> List[Dict[str, np.ndarray]]:
    """Convert a sequence of detections to format of PrecisionRecallF1.

    Args:
        sequence_dets (List[List[fo.Detection]]): A list of fiftyone detections.
        w (int): Width in pixels of the image.
        h (int): Height in pixels of the image.
        is_gt (bool, optional): Flag indicating if the input data is ground truth.
            Defaults to False.
        label_mapping (Dict[str, int], optional): Dictionary mapping string labels to
            numbers, which should be provided if the detection metrics should be
            calculated in a class-specific way. Defaults to None.

    Returns:
        List[Dict[str, np.ndarray]]: A list containing the converted detections.
    """
    output = []

    for frame_dets in sequence_dets:
        frame_dict = frame_dets_to_det_metrics(frame_dets or [], w, h, is_gt, label_mapping)
        output.append(frame_dict)

    return output


def frame_dets_to_det_metrics(
    fo_dets: List[fo.Detection],
    w: int,
    h: int,
    is_gt: bool = False,
    label_mapping: Dict[str, int] = None,
) -> Dict[str, np.ndarray]:
    """Convert a list of fiftyone detections to format of PrecisionRecallF1.

    Args:
        fo_dets (List[fo.Detection]): A list of fiftyone detections.
        w (int): Width in pixels of the image.
        h (int): Height in pixels of the image.
        is_gt (bool, optional): Flag indicating if the input data is ground truth.
            Defaults to False.
        label_mapping (Dict[str, int], optional): Dictionary mapping string labels to
            numbers, which should be provided if the detection metrics should be 
            calculated in a class-specific way. Defaults to None.

    Returns:
        Dict[str, np.ndarray]: A dictionary containing the converted detections.
    """
    global error_code

    detections = []
    labels = []
    scores = []
    areas = []

    for det in fo_dets:
        bbox = det["bounding_box"]
        if not bbox or len(bbox) < 4:
            continue
        if label_mapping and det["label"] not in label_mapping:
            print(f"could not add sample w/ label {det['label']}, \
                  as label is not in label mapping")
            continue

        detections.append([
            bbox[0] * w,
            bbox[1] * h,
            (bbox[0] + bbox[2]) * w,
            (bbox[1] + bbox[3]) * h,
        ])
        labels.append(0 if label_mapping is None else label_mapping[det["label"]])
        scores.append(det["confidence"] if det["confidence"] else 1.0)  # None for gt

        if is_gt:
            if "area" in det.field_names:
                areas.append(det["area"])
            else:
                areas.append(bbox[2] * w * bbox[3] * h)
                if error_code is None:
                    print("⚠️WARNING: Area not found in ground truth annotation(s), \
                          using bbox area instead for these cases.")
                    error_code = 1
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
    img_w,
    img_h,
):
    """Returns:
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
        img_w,
        img_h,
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
        img_w,
        img_h,
    )

    if _TORCHMETRICS_AVAILABLE:
        target, preds = _to_tm_format(target, preds)

    return target, preds


def get_relevant_fields(
    view: fo.DatasetView,
    fields: list,  # fiftyone field names
):
    """Returns a view with only the relevant fields to prevent memory issues.

    Parameters
    ----------
    view: fo.DatasetView
        Dataset view
    fields: list
        List of fiftyone field names. You can use dot notation (embedded.field.name).

    Returns:
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
    """Parameters
    ----------
    view: fo.DatasetView
        Dataset view
    field_name: str
        Fiftyone field name. You can use dot notation (embedded.field.name).

    Returns:
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


@deprecated(reason="⚠️ Output not tested. Use at your own risk.")
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
    image dataset, it computes the metric in a single pass.
    """
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


@deprecated(reason="We do not guarantee the correctness of this function.")
def get_target_and_preds(
    view: fo.DatasetView,
    gt_field: str,  # fiftyone field name
    pred_field: str,  # fiftyone field name
):
    view = get_relevant_fields(view, [gt_field, pred_field])

    img_w = view.first()["metadata"]["width"]
    img_h = view.first()["metadata"]["height"]
    print(f"Resolution: {img_w}x{img_h}")

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
        img_w,
        img_h,
    )

    return target, preds


@deprecated(reason="⚠️ Output not tested. Use at your own risk.")
def compute_metrics(
    view: fo.DatasetView,
    gt_field: str,  # fiftyone field name
    pred_field: str,  # fiftyone field name
    metric_fn: callable,  # torchmetrics metric
    metric_kwargs: dict,
):  # kwargs for metric_fn
    """Computes metrics for a given dataset view."""
    view = get_relevant_fields(view, [gt_field, pred_field])
    img_w = view.first()["metadata"]["width"]
    img_h = view.first()["metadata"]["height"]
    print(f"Resolution: {img_w}x{img_h}")

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
        img_w,
        img_h,
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
    if not pathlib.Path(csv_dirpath).exists():
        pathlib.Path(csv_dirpath).mkdir(parents=True)
    df = sequence_results_to_df(sequence_results)
    df.to_csv(csv_path, index=False)


def get_confidence_metric_vals(
    cocoeval: np.ndarray, T: int, R: int, K: int, A: int, M: int
):
    """Get confidence values for plotting.

    - recall vs confidence
    - precision vs confidence
    - f1-score vs confidence.

    Arguments:
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

    Returns:
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

    return {"conf": x, "precision": p, "recall": r, "f1": f1}


def box_denormalize(boxes: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Denormalizes boxes from [0, 1] to [0, img_w] and [0, img_h].

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


def _filter_det_valid_sequences(
    sequence_list: list,
    view: fo.DatasetView,
    pred_fields: list,
    instances: dict,
) -> list:
    """Filter sequences that have keyframe data for all prediction fields.

    Sequences missing keyframes for any prediction field are logged as failed
    on every metric instance and excluded from the returned list.

    Parameters
    ----------
    sequence_list : list
        Candidate sequence names.
    view : fo.DatasetView
        FiftyOne dataset view used to match individual sequences.
    pred_fields : list
        Prediction field names to validate.
    instances : dict
        Nested dict ``{pred_field: {metric_name: metric_instance}}``.

    Returns:
    -------
    list
        Sequence names where all prediction fields have at least one keyframe.
    """
    def _has_keyframes(seq_view: fo.DatasetView, pred_field: str) -> bool:
        """Return True if any frame in *seq_view* has a truthy keyframe value."""
        try:
            kf_vals = seq_view.values(f"frames[].{pred_field}.keyframe")
            return any(kf for kf in kf_vals if kf)
        except (ValueError, AttributeError, RuntimeError, TypeError, KeyError):
            return False

    valid = []
    for sequence_name in tqdm(sequence_list, desc="Validating sequences"):
        sequence_view = view.match(F("sequence") == sequence_name)
        missing = [pf for pf in pred_fields if not _has_keyframes(sequence_view, pf)]
        if missing:
            exc = ValueError(f"No keyframe data for: {missing}")
            for pf in pred_fields:
                for instance in instances[pf].values():
                    instance.log_failed_sequence(sequence_name, [], [], exc=exc)
        else:
            valid.append(sequence_name)
    return valid


def compute_all_metrics_by_sequence(  # noqa: C901, PLR0912, PLR0914
    view: fo.DatasetView,
    gt_field: str,
    pred_fields: "str | list",
    metrics: list,
    sequence_list: Optional[list] = None,
    keyframe_only: bool = False,
) -> dict:
    """Run multiple detection metrics across multiple prediction fields in one pass.

    Parameters
    ----------
    view : fo.DatasetView
        FiftyOne dataset view to evaluate.  For grouped datasets, slice
        selection should be applied by the caller before passing the view
        (e.g. ``view.select_group_slices(["thermal_wide"])``).
    gt_field : str
        FiftyOne field name for ground-truth detections.
    pred_fields : str or list
        One or more FiftyOne prediction field names.  Pass a string for a
        single model or a list to evaluate multiple models in the same pass.
    metrics : list
        List of ``(metric_fn, metric_kwargs)`` tuples, e.g.
        ``[(DetectionMetrics, {"iou_threshold": 0.5})]``.
    sequence_list : list, optional
        Restrict evaluation to these sequence names.  Defaults to all
        sequences found in the view.
    keyframe_only : bool
        When ``True``, restrict evaluation to frames whose
        ``<pred_field>.keyframe`` flag is truthy.  Sequences with no keyframe
        data for any prediction field are logged to ``failed_sequences`` and
        skipped.  Useful for video datasets where the model only annotates
        keyframes.  Default is ``False`` (all frames evaluated).

    Returns:
    -------
    dict
        Nested dict of the form ``{pred_field: {metric_class_name: metric_instance}}``.

    Raises:
    ------
    ValueError
        If duplicate metric class names are found in *metrics*.

    Example:
    -------
    results = compute_all_metrics_by_sequence(
        view=view,
        gt_field="ground_truth_det",
        pred_fields=["model_a", "model_b"],
        metrics=[(DetectionMetrics, {"iou_threshold": 0.5})],
        keyframe_only=True,
    )
    df = det_metrics_to_df(results["model_a"]["DetectionMetrics"])
    """
    if isinstance(pred_fields, str):
        pred_fields = [pred_fields]

    if view.media_type == "group":
        raise ValueError(
            "Grouped dataset passed directly — apply slice selection before "
            "calling this function (e.g. view.select_group_slices(['thermal_wide']))."
        )

    metric_names = [fn.__name__ for fn, _ in metrics]
    if len(metric_names) != len(set(metric_names)):
        raise ValueError(
            f"Duplicate metric class names in metrics list: {metric_names}. "
            "Each metric class may only appear once."
        )

    resolved: list = (
        list(
            get_relevant_fields(view, [gt_field, *pred_fields, "sequence"]).distinct(
                "sequence"
            )
        )
        if sequence_list is None
        else sequence_list
    )

    instances = {
        pred_field: {fn.__name__: fn(**kwargs) for fn, kwargs in metrics}
        for pred_field in pred_fields
    }

    if keyframe_only:
        valid_sequences = _filter_det_valid_sequences(
            resolved, view, pred_fields, instances
        )
    else:
        valid_sequences = resolved

    for sequence_name in tqdm(valid_sequences, desc="Computing metrics"):
        seq_view = view.match(F("sequence") == sequence_name)
        sample = seq_view.first()
        if sample is None:
            continue

        if seq_view.media_type in {"video", "group"}:
            img_w = sample["metadata"]["frame_width"]
            img_h = sample["metadata"]["frame_height"]
        else:
            img_w = sample["metadata"]["width"]
            img_h = sample["metadata"]["height"]

        gt_frame_dets = get_values(seq_view, f"{gt_field}.detections")

        for pred_field in pred_fields:
            pred_frame_dets = get_values(seq_view, f"{pred_field}.detections")

            if keyframe_only:
                keyframes = get_values(seq_view, f"{pred_field}.keyframe")
                gt_filtered = [
                    dets
                    for kf, dets in zip(keyframes, gt_frame_dets, strict=False)
                    if kf
                ]
                pred_filtered = [
                    dets
                    for kf, dets in zip(keyframes, pred_frame_dets, strict=False)
                    if kf
                ]
            else:
                gt_filtered = gt_frame_dets
                pred_filtered = pred_frame_dets

            targets = payload_sequence_to_det_metrics(
                sequence_dets=gt_filtered,
                w=img_w,
                h=img_h,
                is_gt=True,
            )
            preds = payload_sequence_to_det_metrics(
                sequence_dets=pred_filtered,
                w=img_w,
                h=img_h,
                is_gt=False,
            )

            for instance in instances[pred_field].values():
                instance.update(preds, targets, sequence_name)

    # Synchronise failures: a sequence that failed for one pred_field is
    # marked failed across all pred_fields so det_metrics_to_df returns
    # the same sequence set regardless of which model is requested.
    all_failed: set = {
        seq
        for pf_instances in instances.values()
        for instance in pf_instances.values()
        for seq in instance.failed_sequences
    }
    for pf_instances in instances.values():
        for instance in pf_instances.values():
            for seq in all_failed - set(instance.failed_sequences):
                instance.log_failed_sequence(seq, [], [], exc=None)

    return instances


def det_metrics_to_df(
    metrics: "DetectionMetrics",
    sequence_list: Optional[list] = None,
    area_range_label: str = "all",
) -> pd.DataFrame:
    """Convert a DetectionMetrics instance to a per-sequence DataFrame.

    Parameters
    ----------
    metrics : DetectionMetrics
        Fitted instance returned by ``compute_all_metrics_by_sequence``.
    sequence_list : list, optional
        Sequences to include.  Defaults to all successful accumulators
        (``metrics.accumulators.keys()``).
    area_range_label : str
        Which area range to extract from each sequence result.  Must match
        one of the ``area_ranges_labels`` configured on the underlying
        ``PrecisionRecallF1Support`` (default ``"all"``).

    Returns:
    -------
    pd.DataFrame
        One row per sequence with columns:
        ``sequence``, ``precision``, ``recall``, ``f1``,
        ``tp``, ``fp``, ``fn``, ``duplicates``, ``support``, ``fpi``, ``n_imgs``.

    Example:
    -------
    df_all   = det_metrics_to_df(inst, area_range_label="all")
    df_small = det_metrics_to_df(inst, area_range_label="small")
    """
    if sequence_list is None:
        sequence_list = list(metrics.accumulators.keys())

    rows = []
    for sequence in sequence_list:
        area_results = metrics.compute(sequence=sequence)
        row = area_results.get(area_range_label, {}).copy()
        row["sequence"] = sequence
        rows.append(row)

    return pd.DataFrame(rows)
