"""Utility helpers for building MOT/HOTA metric inputs from FiftyOne views."""

import contextlib
import io
import pathlib

import fiftyone as fo
import numpy as np
import pandas as pd
from fiftyone import ViewField as F
from tqdm import tqdm


def prepare_data_for_det_metrics(  # noqa: C901
    gt_bboxes_per_frame: list,
    gt_track_ids_per_frame: list,
    dt_bboxes_per_frame: list,
    dt_track_ids_per_frame: list,
    dt_scores_per_frame: list,
    img_w: int = 640,
    img_h: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert per-frame detection lists into tracker-format numpy arrays.

    Args:
        gt_bboxes_per_frame: Per-frame list of ground-truth bounding boxes.
        gt_track_ids_per_frame: Per-frame list of ground-truth track IDs.
        dt_bboxes_per_frame: Per-frame list of predicted bounding boxes.
        dt_track_ids_per_frame: Per-frame list of predicted track IDs.
        dt_scores_per_frame: Per-frame list of detection confidence scores.
        img_w: Image width in pixels used for denormalization.
        img_h: Image height in pixels used for denormalization.

    Returns:
        Tuple of (target, preds) numpy arrays in MOT tracker format.
    """

    def _to_tracker_format(
        gt_bboxes_per_frame: list,
        gt_track_ids_per_frame: list,
        dt_bboxes_per_frame: list,
        dt_track_ids_per_frame: list,
        dt_scores_per_frame: list,
        img_w: int,
        img_h: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert a list of frames with detections (bboxes) to numpy format."""
        target = []
        preds = []

        for idx, (bbox, track_id) in enumerate(
            zip(gt_bboxes_per_frame, gt_track_ids_per_frame, strict=False)
        ):
            if bbox is not None:
                for bb, t_id in zip(bbox, track_id, strict=False):
                    denormalized_box = box_convert(
                        box_denormalize(np.array(bb), img_w, img_h),
                        in_fmt="xywh",
                        out_fmt="xyxy",
                    )
                    if t_id is not None:
                        target.append(
                            [
                                idx + 1,
                                t_id,
                                denormalized_box[0],
                                denormalized_box[1],
                                denormalized_box[2],
                                denormalized_box[3],
                                1,
                                -1,
                                -1,
                                -1,
                            ]
                        )

        for idx, (bbox, track_id, score) in enumerate(
            zip(
                dt_bboxes_per_frame,
                dt_track_ids_per_frame,
                dt_scores_per_frame,
                strict=False,
            )
        ):
            if bbox is not None:
                for bb, t_id, s in zip(bbox, track_id, score, strict=False):
                    denormalized_box = box_convert(
                        box_denormalize(np.array(bb), img_w, img_h),
                        in_fmt="xywh",
                        out_fmt="xyxy",
                    )
                    preds.append(
                        [
                            idx + 1,
                            t_id,
                            denormalized_box[0],
                            denormalized_box[1],
                            denormalized_box[2],
                            denormalized_box[3],
                            s,
                            -1,
                            -1,
                            -1,
                        ]
                    )

        return np.array(target), np.array(preds)

    def _validate_arrays(data: list | None, data_type: str) -> list:
        """Validate and normalise per-frame annotation arrays.

        Args:
            data: Raw per-frame annotation data (bboxes, masks, scores or labels).
            data_type: One of ``"bbox"``, ``"mask"``, ``"score"``, or ``"label"``.

        Returns:
            List of numpy arrays, one per frame, with ``None`` frames replaced by
            empty arrays.

        Raises:
            ValueError: If ``data_type`` is not a supported value.
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
        data = [np.array(x) if x is not None else np.array([]) for x in data]
        return data

    def _not_falsy(x: object) -> bool:
        """Return True when *x* is a non-empty, non-None value.

        Args:
            x: Value to test.

        Returns:
            False when *x* is ``None``, an empty list/tuple, or an empty ndarray;
            True otherwise.
        """
        if x is None:
            return False
        if isinstance(x, (list, tuple)) and len(x) == 0:
            return False
        return not (isinstance(x, np.ndarray) and x.size == 0)

    target, preds = _to_tracker_format(
        gt_bboxes_per_frame,
        gt_track_ids_per_frame,
        dt_bboxes_per_frame,
        dt_track_ids_per_frame,
        dt_scores_per_frame,
        img_w=img_w,
        img_h=img_h,
    )

    return target, preds


def get_relevant_fields(
    view: fo.DatasetView,
    fields: list,
) -> fo.DatasetView:
    """Return a view with only the relevant fields to prevent memory issues.

    Args:
        view: Dataset view.
        fields: List of fiftyone field names. You can use dot notation
            (embedded.field.name).

    Returns:
        Dataset view with only the relevant fields.

    Raises:
        ValueError: If the media type of *view* is not ``"video"``.
    """
    if view.media_type == "group":
        view = view.select_group_slices(view.default_group_slice)

    if view.media_type == "video":
        return view.select_fields(
            [f"frames.{f}" if view.has_frame_field(f) else f for f in fields]
        )
    else:
        raise ValueError(f"Unsupported media type: {view.media_type}")


def get_values(
    view: fo.DatasetView,
    field_name: str,
) -> list:
    """Return field values from a FiftyOne view.

    Args:
        view: Dataset view.
        field_name: Fiftyone field name. You can use dot notation
            (embedded.field.name).

    Returns:
        List of values.

    Raises:
        ValueError: If the media type of *view* is not ``"video"`` or ``"image"``.
    """
    if view.media_type == "video":
        return view.values(f"frames[].{field_name}")
    elif view.media_type == "image":
        return view.values(field_name)
    else:
        raise ValueError(f"Unsupported media type: {view.media_type}")


def build_detection_inputs(
    view: fo.DatasetView,
    gt_field: str,
    pred_field: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (target, preds) numpy arrays for the given sequence view.

    Args:
        view: FiftyOne dataset view for the sequence.
        gt_field: FiftyOne field name for ground-truth detections.
        pred_field: FiftyOne field name for predicted detections.

    Returns:
        Tuple of (target, preds) numpy arrays in MOT tracker format.

    Raises:
        ValueError: If the view contains no samples after field selection.
    """
    view = get_relevant_fields(view, [gt_field, pred_field])

    sample = view.first()
    if sample is None:
        raise ValueError("View is empty — no samples found after field selection.")
    img_w = sample["metadata"]["frame_width"]
    img_h = sample["metadata"]["frame_height"]

    gt_bboxes_per_frame = get_values(view, f"{gt_field}.detections.bounding_box")
    gt_track_ids_per_frame = get_values(view, f"{gt_field}.detections.index")
    dt_bboxes_per_frame = get_values(view, f"{pred_field}.detections.bounding_box")
    dt_scores_per_frame = get_values(view, f"{pred_field}.detections.confidence")
    dt_track_ids_per_frame = get_values(view, f"{pred_field}.detections.index")

    keyframes = get_values(view, f"{pred_field}.keyframe")
    gt_bboxes_per_frame = [
        bboxes
        for (kf, bboxes) in zip(keyframes, gt_bboxes_per_frame, strict=False)
        if kf
    ]
    gt_track_ids_per_frame = [
        track_ids
        for (kf, track_ids) in zip(keyframes, gt_track_ids_per_frame, strict=False)
        if kf
    ]
    dt_bboxes_per_frame = [
        bboxes
        for (kf, bboxes) in zip(keyframes, dt_bboxes_per_frame, strict=False)
        if kf
    ]
    dt_scores_per_frame = [
        scores
        for (kf, scores) in zip(keyframes, dt_scores_per_frame, strict=False)
        if kf
    ]
    dt_track_ids_per_frame = [
        track_ids
        for (kf, track_ids) in zip(keyframes, dt_track_ids_per_frame, strict=False)
        if kf
    ]

    target, preds = prepare_data_for_det_metrics(
        gt_bboxes_per_frame,
        gt_track_ids_per_frame,
        dt_bboxes_per_frame,
        dt_track_ids_per_frame,
        dt_scores_per_frame,
        img_w=img_w,
        img_h=img_h,
    )

    del gt_bboxes_per_frame, gt_track_ids_per_frame
    del dt_bboxes_per_frame, dt_scores_per_frame, dt_track_ids_per_frame

    return target, preds


def compute_metrics(
    view: fo.DatasetView,
    gt_field: str,
    pred_field: str,
    metric_fn: callable,
    metric_kwargs: dict,
) -> dict:
    """Compute metrics for a given sequence view.

    Args:
        view: FiftyOne dataset view for the sequence.
        gt_field: FiftyOne field name for ground-truth detections.
        pred_field: FiftyOne field name for predicted detections.
        metric_fn: Metric class constructor (e.g. a torchmetrics metric).
        metric_kwargs: Keyword arguments forwarded to ``metric_fn``.

    Returns:
        Dictionary returned by ``metric.compute()``.
    """
    target, preds = build_detection_inputs(view, gt_field, pred_field)
    metric = metric_fn(**metric_kwargs)
    metric.update(preds, target)
    return metric.compute()


def sequence_results_to_df(sequence_results: dict) -> pd.DataFrame:
    """Convert a sequence-results dict to a pandas DataFrame.

    Args:
        sequence_results: Nested dict of the form
            ``{seq_name: {"metrics": {area_range_lbl: metric_dict}}}``.

    Returns:
        DataFrame with one row per (sequence, area_range_lbl) combination.
    """
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


def compute_and_save_sequence_metrics(
    csv_dirpath: str,
    view: fo.DatasetView,
    gt_field: str,
    pred_field: str,
    metric_fn: callable,
    metric_kwargs: dict,
    csv_suffix: str | None = None,
    debug: bool = False,
    name_separator: str = "__",
) -> None:
    """Compute per-sequence metrics and save results to a CSV file.

    Args:
        csv_dirpath: Directory where the output CSV will be written.
        view: FiftyOne dataset view to evaluate.
        gt_field: FiftyOne field name for ground-truth detections.
        pred_field: FiftyOne field name for predicted detections.
        metric_fn: Metric class constructor (e.g. a torchmetrics metric).
        metric_kwargs: Keyword arguments forwarded to ``metric_fn``.
        csv_suffix: Optional suffix appended to the generated CSV filename.
        debug: When True, print captured stdout for each sequence.
        name_separator: String used to join CSV filename components.
    """
    csv_name = name_separator.join(
        [view.dataset_name, gt_field, pred_field, metric_fn.__name__]
    )
    csv_name = name_separator.join([csv_name, csv_suffix]) if csv_suffix else csv_name
    csv_name += ".csv"
    csv_path = str(pathlib.Path(csv_dirpath) / csv_name)
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

    if not pathlib.Path(csv_dirpath).exists():
        pathlib.Path(csv_dirpath).mkdir(parents=True)
    df = sequence_results_to_df(sequence_results)
    df.to_csv(csv_path, index=False)


def compute_metrics_by_sequence(
    view: fo.DatasetView,
    gt_field: str,
    pred_field: str,
    metric_fn: callable,
    metric_kwargs: dict,
    sequence_list: list | None = None,
) -> object:
    """Compute a single metric across all sequences in a view.

    Args:
        view: FiftyOne dataset view to evaluate.
        gt_field: FiftyOne field name for ground-truth detections.
        pred_field: FiftyOne field name for predicted detections.
        metric_fn: Metric class constructor.
        metric_kwargs: Keyword arguments forwarded to ``metric_fn``.
        sequence_list: Optional list of sequence names to restrict evaluation.
            Defaults to all sequences found in the view.

    Returns:
        Fitted metric instance after calling ``update`` on every sequence.
    """
    sequence_results = {}

    metric = metric_fn(**metric_kwargs)
    resolved_sequences: list = (
        sequence_list
        if sequence_list is not None
        else list(
            get_relevant_fields(view, [gt_field, pred_field, "sequence"]).distinct(
                "sequence"
            )
        )
    )
    for sequence_name in resolved_sequences:
        sequence_view = view.match(F("sequence") == sequence_name)
        sequence_results[sequence_name] = build_detection_inputs(
            view=sequence_view, gt_field=gt_field, pred_field=pred_field
        )
    for sequence in sequence_results:
        try:
            metric.update(
                sequence_results[sequence][0], sequence_results[sequence][1], sequence
            )
        except (ValueError, IndexError) as e:
            metric.log_failed_sequence(
                sequence,
                sequence_results[sequence][0],
                sequence_results[sequence][1],
                exc=e,
            )
        except Exception:
            raise

    return metric


def compute_all_metrics_by_sequence(  # noqa: C901,PLR0912
    view: fo.DatasetView,
    gt_field: str,
    pred_fields: "str | list",
    metrics: list,
    sequence_list: list | None = None,
) -> dict:
    """Run multiple metrics across multiple prediction fields in a single pass.

    Args:
        view: FiftyOne dataset view to evaluate.
        gt_field: FiftyOne field name for ground-truth detections.
        pred_fields: One or more FiftyOne prediction field names. Pass a string
            for a single model or a list to evaluate multiple models in the same
            pass.
        metrics: List of (metric_fn, metric_kwargs) tuples, e.g.
            ``[(TrackingMetrics, {"max_iou": 0.5}), (HOTAMetrics, {})]``.
        sequence_list: Optional list of sequence names to restrict evaluation.
            Defaults to all sequences found in the view.

    Returns:
        Nested dict of the form ``{pred_field: {metric_class_name: metric_instance}}``.

    Example:
        results = compute_all_metrics_by_sequence(
            view=view,
            gt_field="ground_truth_det",
            pred_fields=["model_a", "model_b"],
            metrics=[(TrackingMetrics, {"max_iou": 0.5}), (HOTAMetrics, {})],
        )
        mot_df = results_to_df(results["model_a"]["TrackingMetrics"])
    """
    if isinstance(pred_fields, str):
        pred_fields = [pred_fields]

    if sequence_list is None:
        sequence_list = get_relevant_fields(
            view, [gt_field, *pred_fields, "sequence"]
        ).distinct("sequence")

    metric_names = [fn.__name__ for fn, _ in metrics]
    if len(metric_names) != len(set(metric_names)):
        raise ValueError(
            f"Duplicate metric class names in metrics list: {metric_names}. "
            "Each metric class may only appear once."
        )

    instances = {
        pred_field: {fn.__name__: fn(**kwargs) for fn, kwargs in metrics}
        for pred_field in pred_fields
    }

    def _has_keyframes(seq_view: fo.DatasetView, pred_field: str) -> bool:
        """Return True if any frame in *seq_view* has keyframe data for *pred_field*.

        Args:
            seq_view: FiftyOne view for a single sequence.
            pred_field: Prediction field name to check.

        Returns:
            True if at least one keyframe value is truthy; False otherwise.
        """
        video_view = (
            seq_view.select_group_slices(seq_view.default_group_slice)
            if seq_view.media_type == "group"
            else seq_view
        )
        try:
            kf_vals = video_view.values(f"frames[].{pred_field}.keyframe")
            return any(kf for kf in kf_vals if kf)
        except Exception:
            return False

    valid_sequences = []
    for sequence_name in tqdm(sequence_list, desc="Validating sequences"):
        sequence_view = view.match(F("sequence") == sequence_name)
        missing = [pf for pf in pred_fields if not _has_keyframes(sequence_view, pf)]
        if missing:
            exc = ValueError(f"No keyframe data for: {missing}")
            for pf in pred_fields:
                for instance in instances[pf].values():
                    instance.log_failed_sequence(sequence_name, [], [], exc=exc)
        else:
            valid_sequences.append(sequence_name)

    for sequence_name in tqdm(valid_sequences, desc="Computing metrics"):
        sequence_view = view.match(F("sequence") == sequence_name)
        for pred_field in tqdm(pred_fields, desc="Models", leave=False):
            gt, pred = build_detection_inputs(
                view=sequence_view, gt_field=gt_field, pred_field=pred_field
            )
            for instance in instances[pred_field].values():
                try:
                    instance.update(gt, pred, sequence_name)
                except (ValueError, IndexError) as e:
                    instance.log_failed_sequence(sequence_name, gt, pred, exc=e)
                except Exception:
                    raise

    return instances


def compute_sizes(view: fo.DatasetView, gt_field: str) -> list:
    """Compute bounding-box areas for all annotated objects in a sequence view.

    Args:
        view: FiftyOne dataset view for the sequence.
        gt_field: FiftyOne field name for ground-truth detections.

    Returns:
        List of ``[frame_idx, track_id, area]`` entries for every annotated object.

    Raises:
        ValueError: If the view contains no samples after field selection.
    """
    view = get_relevant_fields(view, [gt_field])
    sample = view.first()
    if sample is None:
        raise ValueError("View is empty — no samples found after field selection.")
    img_w = sample["metadata"]["frame_width"]
    img_h = sample["metadata"]["frame_height"]
    gt_bboxes_per_frame = get_values(view, f"{gt_field}.detections.bounding_box")
    gt_track_ids_per_frame = get_values(view, f"{gt_field}.detections.index")

    b = [
        (bboxes, t_ids)
        for (bboxes, t_ids) in zip(
            gt_bboxes_per_frame, gt_track_ids_per_frame, strict=False
        )
        if bboxes is not None and t_ids is not None
    ]
    gt_bboxes_per_frame = [bboxes for (bboxes, _) in b]
    gt_track_ids_per_frame = [t_ids for (_, t_ids) in b]
    objects = []
    for idx, (bbox, t_id) in enumerate(
        zip(gt_bboxes_per_frame, gt_track_ids_per_frame, strict=False)
    ):
        if bbox is not None:
            for bb, track_id in zip(bbox, t_id, strict=False):
                denormalized_box = box_denormalize(np.array(bb), img_w, img_h)
                objects.append(
                    [idx, track_id, denormalized_box[2] * denormalized_box[3]]
                )

    del gt_bboxes_per_frame, gt_track_ids_per_frame

    return objects


def get_sequence_info(
    view: fo.DatasetView,
    gt_field: str,
    sequence_list: list | None = None,
) -> dict:
    """Collect per-object size information grouped by sequence.

    Args:
        view: FiftyOne dataset view to evaluate.
        gt_field: FiftyOne field name for ground-truth detections.
        sequence_list: Optional list of sequence names to restrict evaluation.
            Defaults to all sequences found in the view.

    Returns:
        Dict mapping each sequence name to the list returned by
        :func:`compute_sizes`.
    """
    sequence_info = {}
    sequence_names = get_relevant_fields(view, [gt_field, "sequence"]).distinct(
        "sequence"
    )
    if sequence_list is None:
        sequence_list = sequence_names

    for sequence_name in tqdm(sequence_list):
        sequence_view = view.match(F("sequence") == sequence_name)
        sequence_info[sequence_name] = compute_sizes(
            view=sequence_view,
            gt_field=gt_field,
        )

    return sequence_info


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
    x, y, w, h = np.split(boxes, 4, axis=-1)
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    converted_boxes = np.concatenate([x1, y1, x2, y2], axis=-1)
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
    cx, cy, w, h = np.split(boxes, 4, axis=-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    converted_boxes = np.concatenate([x1, y1, x2, y2], axis=-1)
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
    x1, y1, x2, y2 = np.split(boxes, 4, axis=-1)
    w = x2 - x1
    h = y2 - y1
    converted_boxes = np.concatenate([x1, y1, w, h], axis=-1)
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
    x1, y1, x2, y2 = np.split(boxes, 4, axis=-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    converted_boxes = np.concatenate([cx, cy, w, h], axis=-1)
    return converted_boxes


def results_to_df(metrics: object, sequence_list: list | None = None) -> pd.DataFrame:
    """Convert TrackingMetrics or HOTAMetrics results to a DataFrame.

    Detects the metric type from the result keys and applies metric-specific
    scaling where implemented.

    TrackingMetrics: only ``mota`` is scaled x100 and ``motp`` is converted to
    ``(1 - motp) x 100``; all other returned metrics are left unchanged.
    HOTAMetrics: all returned metric values (hota, deta, assa, loca) are scaled x100.

    Args:
        metrics: Fitted metric instance exposing ``accumulators`` and
            ``compute(sequence=...)``.
        sequence_list: Optional list of sequence names to include. Defaults to
            all accumulators in *metrics*.

    Returns:
        DataFrame with one row per sequence and one column per metric value.
    """
    if sequence_list is None:
        sequence_list = list(metrics.accumulators.keys())  # type: ignore[attr-defined]

    rows = []
    for sequence in sequence_list:
        result = metrics.compute(sequence=sequence)  # type: ignore[attr-defined]

        if "hota" in result:
            row = {
                k: (v if k == "num_unique_objects" else v * 100)
                for k, v in result.items()
            }
        else:
            row = {k: next(iter(v.values())) for k, v in result.items()}
            row["mota"] *= 100
            row["motp"] = (1 - row["motp"]) * 100

        row["sequence"] = sequence
        rows.append(row)

    return pd.DataFrame(rows)


def hota_results_to_df(
    metrics: object, sequence_list: list | None = None
) -> pd.DataFrame:
    """Alias for results_to_df for backward compatibility.

    Args:
        metrics: Fitted metric instance (see :func:`results_to_df`).
        sequence_list: Optional list of sequence names to include.

    Returns:
        DataFrame with one row per sequence and one column per metric value.
    """
    return results_to_df(metrics, sequence_list)


def classify_num_objects(x: int | float) -> str | None:
    """Map an object count to a human-readable category label.

    Args:
        x: Number of objects in a frame or sequence.

    Returns:
        One of ``"zero"``, ``"one"``, ``"two"``, ``"few"``, ``"many"``; or
        ``None`` if *x* does not fall within any defined range.
    """
    n_objects_ranges_tuples = [
        ("zero", [0, 1]),
        ("one", [1, 2]),
        ("two", [2, 3]),
        ("few", [3, 7]),
        ("many", [7, 20]),
    ]
    category = None
    for label, n_objects_range in n_objects_ranges_tuples:
        if n_objects_range[0] <= x < n_objects_range[1]:
            category = label
            break
    return category
