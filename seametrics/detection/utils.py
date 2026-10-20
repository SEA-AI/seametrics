"""Convert FiftyOne and payload detections to metric inputs, and report results."""

import contextlib
import io
import pathlib
from itertools import compress
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import fiftyone as fo
import numpy as np
import pandas as pd
from deprecated import deprecated
from fiftyone import ViewField as F
from tqdm import tqdm

from seametrics.detection.np.utils import box_convert
from seametrics.detection.pooling import OVERALL_KEY, aggregate_sequence_results
from seametrics.payload import Payload

# payload functions


def _sequence_det_inputs(
    payload: Payload,
    sequence_name: str,
    model_name: str,
    label_mapping: Optional[Dict[str, int]],
    keyframes_only: bool,
) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]]]:
    """Convert one sequence's predictions and ground truth to metric inputs.

    Args:
        payload (Payload): The payload holding the sequence.
        sequence_name (str): Name of the sequence.
        model_name (str): Prediction field to read.
        label_mapping (Dict[str, int], optional): Label to class-id mapping;
            ``None`` for class-agnostic metrics.
        keyframes_only (bool): Keep only the frames *model_name* flagged as
            keyframes, in both the predictions and the ground truth.

    Returns:
        Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]]]:
            ``(predictions, references)``, one dict per frame.

    Raises:
        ValueError: If *keyframes_only* is set and the sequence has no keyframe
            mask for *model_name*, or one whose length differs from the frames.
    """
    sequence = payload.sequences[sequence_name]
    pred_frames = sequence[model_name]
    gt_frames = sequence[payload.gt_field_name]
    if keyframes_only:
        mask = (getattr(sequence, "keyframes", None) or {}).get(model_name)
        if mask is None:
            raise ValueError(
                f"Sequence {sequence_name!r} has no keyframe data for {model_name!r}."
                " Rebuild the payload with a PayloadProcessor that records keyframes,"
                " or pass keyframes_only=False."
            )
        if not len(mask) == len(pred_frames) == len(gt_frames):
            raise ValueError(
                f"Keyframe mask for sequence {sequence_name!r} has {len(mask)} entries"
                f" but the sequence has {len(pred_frames)} prediction and"
                f" {len(gt_frames)} ground-truth frames."
            )
        pred_frames = list(compress(pred_frames, mask))
        gt_frames = list(compress(gt_frames, mask))
    w, h = sequence.resolution.width, sequence.resolution.height
    predictions = payload_sequence_to_det_metrics(
        sequence_dets=pred_frames, w=w, h=h, label_mapping=label_mapping
    )
    references = payload_sequence_to_det_metrics(
        sequence_dets=gt_frames, w=w, h=h, is_gt=True, label_mapping=label_mapping
    )
    return predictions, references


def payload_to_det_metric(
    payload: Payload,
    model_name: Optional[str] = None,
    label_mapping: Optional[Dict[str, int]] = None,
    class_agnostic: bool = True,
    keyframes_only: bool = False,
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
        keyframes_only (bool, optional): Keep only the frames the model flagged as
            keyframes, in ground truth and predictions alike, as
            ``seametrics.tracking`` does. Requires sequences that record
            keyframes. Defaults to False.

    Returns:
        Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]]]:
            A tuple containing the converted (predictions, references), with
            absolute ``xywh`` boxes: create the metric with ``box_format="xywh"``.
    """
    if class_agnostic and label_mapping is not None:
        raise ValueError("Label mapping cannot be provided for class-agnostic metrics.")

    predictions, references = [], []

    if model_name is None:
        model_name = payload.models[0]

    for sequence_name in payload.sequences:
        sequence_predictions, sequence_references = _sequence_det_inputs(
            payload, sequence_name, model_name, label_mapping, keyframes_only
        )
        predictions.extend(sequence_predictions)
        references.extend(sequence_references)

    return predictions, references


def payload_to_det_metrics_by_sequence(
    payload: Payload,
    model_name: Optional[str] = None,
    label_mapping: Optional[Dict[str, int]] = None,
    class_agnostic: bool = True,
    *,
    include_overall: bool = True,
    keyframes_only: bool = False,
    **metric_kwargs: object,
) -> Dict[str, dict]:
    """Evaluate every sequence of a payload separately, plus a pooled total.

    One metric over every frame weights precision and recall by support, so one
    long sequence can hide a regression in a short one.
    ``box_format`` is fixed to ``"xywh"``, the layout
    :func:`frame_dets_to_det_metrics` emits.

    Args:
        payload (Payload): The payload containing sequences, models and the
            ground truth field name.
        model_name (str, optional): The name of the model to evaluate. If not
            provided, the first model in the payload is used.
        label_mapping (Dict[str, int], optional): Dictionary mapping string labels
            to numbers, required for class-specific metrics. Defaults to None.
        class_agnostic (bool, optional): Flag indicating if the metrics should be
            calculated in a class-agnostic way. Defaults to True.
        include_overall (bool, optional): Add an ``"OVERALL"`` entry pooled from
            the per-sequence counts. Defaults to True.
        keyframes_only (bool, optional): As for :func:`payload_to_det_metric`.
            Masks are per model, so two models may be scored on different frames.
            Defaults to False.
        **metric_kwargs: Forwarded to ``PrecisionRecallF1Support``. With
            *label_mapping*, ``labels`` defaults to the sorted mapping values so
            every sequence reports the same classes.

    Returns:
        Dict[str, dict]: ``{sequence_name: compute() output}``. ``"OVERALL"``
            comes last and equals one metric over every frame.

    Raises:
        ValueError: If *label_mapping* is combined with ``class_agnostic=True``,
            if ``box_format`` is in *metric_kwargs*, or if a sequence is named
            ``"OVERALL"`` while *include_overall* is set. Errors in a sequence
            propagate; ``seametrics.tracking`` logs them to ``failed_sequences``.
    """
    from seametrics.detection import PrecisionRecallF1Support

    if class_agnostic and label_mapping is not None:
        raise ValueError("Label mapping cannot be provided for class-agnostic metrics.")

    if "box_format" in metric_kwargs:
        raise ValueError(
            "`box_format` cannot be overridden: frame_dets_to_det_metrics always"
            " produces absolute xywh boxes."
        )

    if include_overall and OVERALL_KEY in payload.sequences:
        raise ValueError(
            f"A sequence is named {OVERALL_KEY!r}, which collides with the pooled"
            f" entry. Rename it or pass include_overall=False."
        )

    if model_name is None:
        model_name = payload.models[0]

    if label_mapping is not None:
        metric_kwargs.setdefault("labels", sorted(set(label_mapping.values())))

    sequence_results = {}
    for sequence_name in payload.sequences:
        metric = PrecisionRecallF1Support(
            box_format="xywh",
            class_agnostic=class_agnostic,
            **metric_kwargs,
        )
        predictions, references = _sequence_det_inputs(
            payload, sequence_name, model_name, label_mapping, keyframes_only
        )
        metric.update(predictions, references)
        sequence_results[sequence_name] = metric.compute()

    if include_overall and sequence_results:
        sequence_results[OVERALL_KEY] = aggregate_sequence_results(sequence_results)

    return sequence_results


def payload_sequence_to_det_metrics(
    sequence_dets: List[List[fo.Detection]],
    w: int,
    h: int,
    is_gt: bool = False,
    label_mapping: Optional[Dict[str, int]] = None,
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
        List[Dict[str, np.ndarray]]: A list containing the converted detections,
            one dict per frame, as :func:`frame_dets_to_det_metrics` builds it.
    """
    output = []

    for frame_dets in sequence_dets:
        frame_dict = frame_dets_to_det_metrics(frame_dets, w, h, is_gt, label_mapping)
        output.append(frame_dict)

    return output


def frame_dets_to_det_metrics(
    fo_dets: List[fo.Detection],
    w: int,
    h: int,
    is_gt: bool = False,
    label_mapping: Optional[Dict[str, int]] = None,
) -> Dict[str, np.ndarray]:
    """Convert one frame's fiftyone detections to PrecisionRecallF1Support input.

    FiftyOne boxes are relative ``[x, y, width, height]``; the output keeps that
    layout in pixels, so the metric needs ``box_format="xywh"``.

    Args:
        fo_dets (List[fo.Detection]): A list of fiftyone detections.
        w (int): Width in pixels of the image.
        h (int): Height in pixels of the image.
        is_gt (bool, optional): Flag indicating if the input data is ground truth.
            Defaults to False.
        label_mapping (Dict[str, int], optional): Dictionary mapping string labels to
            numbers, which should be provided if the detection metrics should be
            calculated in a class-specific way. Detections whose label is missing
            from the mapping are dropped. Defaults to None.

    Returns:
        Dict[str, np.ndarray]: ``boxes`` ``(N, 4)`` and ``labels`` ``(N,)``, plus
            ``scores`` for predictions (``1.0`` when confidence is None). Empty
            input yields 1-D empty arrays. No ``area``: the metric derives it from
            the box.
    """
    boxes = []
    labels = []
    scores = []

    for det in fo_dets:
        if label_mapping and det["label"] not in label_mapping:
            print(
                f"could not add sample w/ label {det['label']}, \
                  as label is not in label mapping"
            )
            continue

        x, y, box_w, box_h = det["bounding_box"]
        boxes.append([x * w, y * h, box_w * w, box_h * h])
        labels.append(0 if label_mapping is None else label_mapping[det["label"]])

        if not is_gt:
            scores.append(1.0 if det["confidence"] is None else det["confidence"])

    metrics_dict = {
        "boxes": np.array(boxes),
        "labels": np.array(labels),
    }
    if not is_gt:
        metrics_dict["scores"] = np.array(scores)
    return metrics_dict


# DEPRECATED helper functions
def _frames_to_np(
    bboxes_per_frame: Sequence[np.ndarray],
    labels_per_frame: Sequence[np.ndarray],
    img_size: Tuple[int, int],
    scores_per_frame: Optional[Sequence[Sequence[float]]] = None,
) -> List[Dict[str, np.ndarray]]:
    """Converts per-frame normalized xywh boxes and labels to metric input dicts.

    Args:
        bboxes_per_frame: Normalized ``xywh`` boxes of each frame.
        labels_per_frame: Labels of each frame.
        img_size: ``(width, height)`` of the image in pixels.
        scores_per_frame: Scores of each frame; ``None`` for ground truth.
    """
    img_w, img_h = img_size
    frames = []
    rows = (
        zip(bboxes_per_frame, labels_per_frame, strict=False)
        if scores_per_frame is None
        else zip(bboxes_per_frame, labels_per_frame, scores_per_frame, strict=False)
    )
    for boxes, labels, *scores in rows:
        frame = {
            "boxes": box_convert(
                box_denormalize(boxes, img_w, img_h),
                in_fmt="xywh",
                out_fmt="xyxy",
            ),
            "labels": np.unique(labels, return_inverse=True)[1],
        }
        if scores:
            frame["scores"] = np.array(scores[0])
        frames.append(frame)
    return frames


def _not_falsy(x: object) -> bool:
    """Return False for None and for empty lists, tuples and arrays."""
    if x is None:
        return False
    if isinstance(x, (list, tuple)) and len(x) == 0:
        return False
    return not (isinstance(x, np.ndarray) and x.size == 0)


def _validate_arrays(data: object, data_type: str) -> List[np.ndarray]:
    """Wrap a single frame into a list of frames and convert each to an array.

    Raises:
        ValueError: If ``data_type`` is not one of bbox, mask, score or label.
    """
    if data is None or len(data) == 0:
        data = [data]
    if data_type in {"bbox", "mask"}:
        if any(
            _not_falsy(item) and not isinstance(item[0], (tuple, list, np.ndarray))
            for item in data
        ):
            data = [data]
    elif data_type in {"score", "label"}:
        if any(
            _not_falsy(item) and not isinstance(item, (tuple, list, np.ndarray))
            for item in data
        ):
            data = [data]
    else:
        raise ValueError(f"Unsupported data type: {data_type}")
    return [np.array(x) if x is not None else np.array([]) for x in data]


# Public signature, called positionally from notebooks; kept for compatibility.
def prepare_data_for_det_metrics(  # pylint: disable=too-many-positional-arguments
    gt_bboxes_per_frame: object,
    gt_labels_per_frame: object,
    dt_bboxes_per_frame: object,
    dt_labels_per_frame: object,
    dt_scores_per_frame: object,
    img_w: int,
    img_h: int,
) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]]]:
    """Converts per-frame normalized detections to metric inputs.

    Returns:
        target, preds: tuple of list of dicts. Each dict has keys "boxes",
        "labels", "scores" (scores is only in preds).
    """
    gt_bboxes_per_frame = _validate_arrays(gt_bboxes_per_frame, data_type="bbox")
    gt_labels_per_frame = _validate_arrays(gt_labels_per_frame, data_type="label")
    dt_bboxes_per_frame = _validate_arrays(dt_bboxes_per_frame, data_type="bbox")
    dt_labels_per_frame = _validate_arrays(dt_labels_per_frame, data_type="label")
    dt_scores_per_frame = _validate_arrays(dt_scores_per_frame, data_type="score")

    assert len(gt_bboxes_per_frame) == len(dt_bboxes_per_frame), (
        "Number of frames in GT and prediction do not match"
        + f" ({len(gt_bboxes_per_frame)} vs {len(dt_bboxes_per_frame)})"
    )

    if all(item is None for sublist in dt_scores_per_frame for item in sublist):
        dt_scores_per_frame = [[1.0] * len(x) for x in dt_scores_per_frame]

    target = _frames_to_np(gt_bboxes_per_frame, gt_labels_per_frame, (img_w, img_h))
    preds = _frames_to_np(
        dt_bboxes_per_frame,
        dt_labels_per_frame,
        (img_w, img_h),
        scores_per_frame=dt_scores_per_frame,
    )
    return target, preds


def get_relevant_fields(
    view: fo.DatasetView,
    fields: list,  # fiftyone field names
) -> fo.DatasetView:
    """Returns a view with only the relevant fields to prevent memory issues.

    Args:
        view: Dataset view
        fields: List of fiftyone field names. You can use dot notation
            (embedded.field.name).

    Returns:
        Dataset view with only the relevant fields.

    Raises:
        ValueError: If the view is neither a video nor an image view.
    """
    if view.media_type == "video":
        return view.select_fields(
            [f"frames.{f}" if view.has_frame_field(f) else f for f in fields]
        )
    if view.media_type == "image":
        return view.select_fields(fields)
    raise ValueError(f"Unsupported media type: {view.media_type}")


def get_values(
    view: fo.DatasetView,
    field_name: str,  # fiftyone field name
) -> list:
    """Returns the values of a field, flattened across frames for videos.

    Args:
        view: Dataset view
        field_name: Fiftyone field name. You can use dot notation
            (embedded.field.name).

    Returns:
        List of values.

    Raises:
        ValueError: If the view is neither a video nor an image view.
    """
    if view.media_type == "video":
        return view.values(f"frames[].{field_name}")
    if view.media_type == "image":
        return view.values(field_name)
    raise ValueError(f"Unsupported media type: {view.media_type}")


@deprecated(reason="⚠️ Output not tested. Use at your own risk.")
# conf_thr is never applied but is part of the signature and passed by notebooks.
def smart_compute_metrics(  # pylint: disable=too-many-positional-arguments
    view: fo.DatasetView,
    metric_fn: Callable,  # metric class, e.g. PrecisionRecallF1Support
    metric_kwargs: dict,
    gt_field: str,  # fiftyone field name
    pred_field: str,  # fiftyone field name
    conf_thr: float = 0,  # ruff: ignore[unused-function-argument]  # pylint: disable=unused-argument
) -> dict:
    """Computes a metric over a view, updating it once per sequence for videos.

    Raises:
        ValueError: If the view is neither a video nor an image view.
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
) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]]]:
    """Fetches a view's ground truth and predictions as metric inputs."""
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
    metric_fn: Callable,  # metric class, e.g. PrecisionRecallF1Support
    metric_kwargs: dict,
) -> dict:
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


def results_to_df(results: dict, fixed_columns: Optional[dict] = None) -> pd.DataFrame:
    """Returns one DataFrame row per area range of a metric result."""
    fixed_columns = fixed_columns or {}
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


def sequence_results_to_df(sequence_results: dict) -> pd.DataFrame:
    """Returns one DataFrame row per sequence and area range."""
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


# Public signature; kept for compatibility.
def compute_and_save_sequence_metrics(  # pylint: disable=too-many-positional-arguments
    csv_dirpath: str,
    view: fo.DatasetView,
    gt_field: str,  # fiftyone field name
    pred_field: str,  # fiftyone field name
    metric_fn: Callable,  # metric class, e.g. PrecisionRecallF1Support
    metric_kwargs: dict,
    csv_suffix: Optional[str] = None,
    debug: bool = False,
    name_separator: str = "__",
) -> None:
    """Computes metrics per sequence and saves them to a csv in ``csv_dirpath``."""
    csv_name = name_separator.join(
        [view.dataset_name, gt_field, pred_field, metric_fn.__name__]
    )
    csv_name = name_separator.join([csv_name, csv_suffix]) if csv_suffix else csv_name
    csv_name += ".csv"
    csv_path = pathlib.Path(csv_dirpath) / csv_name
    print(f"Saving metrics to {csv_path}")

    view = get_relevant_fields(view, [gt_field, pred_field, "sequence"])

    sequence_results = {}
    sequence_names = view.distinct("sequence")
    for sequence_name in tqdm(sequence_names):
        with contextlib.redirect_stdout(io.StringIO()) as f:
            print(sequence_name)
            sequence_results[sequence_name] = compute_metrics(
                view=view.match(F("sequence") == sequence_name),
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
    sequence_results_to_df(sequence_results).to_csv(csv_path, index=False)


# Public signature: notebooks pass T, R, K, A, M by keyword; R is unused.
def get_confidence_metric_vals(  # pylint: disable=too-many-positional-arguments,unused-argument
    cocoeval: np.ndarray,
    T: int,
    R: int,  # ruff: ignore[unused-function-argument]
    K: int,
    A: int,
    M: int,
) -> dict:
    """Get confidence values for plotting.

    - recall vs confidence
    - precision vs confidence
    - f1-score vs confidence.

    Args:
        cocoeval: The ``eval`` dict of a ``compute()`` result.
        T: IoU threshold index.
        R: Recall threshold index; unused.
        K: Category index.
        A: Area range index.
        M: Max detections index.

    Returns:
        dict of ``conf``, a 1000-point grid over [0, 1], and the ``precision``,
        ``recall`` and ``f1`` curves on it.
    """
    tpc = cocoeval["TPC"][T, K, A, M]
    fpc = cocoeval["FPC"][T, K, A, M]
    n_gt = cocoeval["TP"][T, K, A, M] + cocoeval["FN"][T, K, A, M]
    conf = cocoeval["sorted_conf"][K, A, M]
    eps = 1e-16
    x = np.linspace(0, 1, 1000)  # for plotting

    # Recall curve; negative x, xp because xp decreases
    r = np.interp(-x, -conf, tpc / (n_gt + eps), left=0)

    # Precision curve
    p = np.interp(-x, -conf, tpc / (tpc + fpc), left=1)

    # F1-score
    f1 = 2 * p * r / (p + r + eps)

    return {"conf": x, "precision": p, "recall": r, "f1": f1}


def box_denormalize(boxes: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Denormalize boxes from [0, 1] to [0, img_w] and [0, img_h].

    Works for any layout with x-like values in even columns (``xyxy``, ``xywh``,
    ``cxcywh``). Boxes with any value above ``1.0`` are returned untouched.

    Args:
        boxes (np.ndarray): Boxes which will be denormalized, shape ``(N, 4)``.
        img_w (int): Width of image in pixels.
        img_h (int): Height of image in pixels.

    Returns:
        np.ndarray: A float copy of the denormalized boxes, or *boxes* itself when
            empty or already in pixels.
    """
    if boxes.size == 0:
        return boxes

    # check if boxes are normalized
    if np.any(boxes > 1.0):
        return boxes

    # a float copy: callers keep their boxes, and integer input does not truncate
    boxes = boxes.astype(np.float64)
    boxes[:, 0::2] *= img_w
    boxes[:, 1::2] *= img_h
    return boxes
