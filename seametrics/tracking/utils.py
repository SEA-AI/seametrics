"""Utility helpers for building MOT/HOTA metric inputs from FiftyOne views."""

from __future__ import annotations

import contextlib
import io
import pathlib

import numpy as np
import pandas as pd
from tqdm import tqdm

from ._box_utils import box_convert, box_denormalize

try:
    import fiftyone as fo
    from fiftyone import ViewField as F

    _FIFTYONE_AVAILABLE = True
except ImportError:
    _FIFTYONE_AVAILABLE = False


def prepare_data_for_det_metrics(  # noqa: C901
    gt_bboxes_per_frame: list,
    gt_track_ids_per_frame: list,
    dt_bboxes_per_frame: list,
    dt_track_ids_per_frame: list,
    dt_scores_per_frame: list,
    *,
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
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert a list of frames with detections (bboxes) to numpy format.

        Uses ``img_w`` and ``img_h`` from the enclosing function scope.
        """
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
        ImportError: If ``fiftyone`` is not installed.
        ValueError: If the media type of *view* is not ``"video"``.
    """
    if not _FIFTYONE_AVAILABLE:
        raise ImportError("fiftyone is required for this function")
    if view.media_type == "group":
        view = view.select_group_slices(view.default_group_slice)

    if view.media_type == "video":
        return view.select_fields(
            [f"frames.{f}" if view.has_frame_field(f) else f for f in fields]
        )
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
        ImportError: If ``fiftyone`` is not installed.
        ValueError: If the media type of *view* is not ``"video"`` or ``"image"``.
    """
    if not _FIFTYONE_AVAILABLE:
        raise ImportError("fiftyone is required for this function")
    if view.media_type == "video":
        return view.values(f"frames[].{field_name}")
    if view.media_type == "image":
        return view.values(field_name)
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
        ImportError: If ``fiftyone`` is not installed.
        ValueError: If the view contains no samples after field selection.
    """
    if not _FIFTYONE_AVAILABLE:
        raise ImportError("fiftyone is required for this function")
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

    Raises:
        ImportError: If ``fiftyone`` is not installed.
    """
    if not _FIFTYONE_AVAILABLE:
        raise ImportError("fiftyone is required for this function")
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


def _collect_sequence_results(
    view: fo.DatasetView,
    gt_field: str,
    pred_field: str,
    metric_fn: callable,
    metric_kwargs: dict,
    *,
    debug: bool,
) -> dict:
    """Run ``compute_metrics`` over every sequence in *view* and collect results.

    Args:
        view: FiftyOne dataset view (already field-filtered).
        gt_field: FiftyOne field name for ground-truth detections.
        pred_field: FiftyOne field name for predicted detections.
        metric_fn: Metric class constructor.
        metric_kwargs: Keyword arguments forwarded to ``metric_fn``.
        debug: When True, print captured stdout for each sequence.

    Returns:
        Dict mapping sequence name to its ``compute_metrics`` result dict.

    Raises:
        ImportError: If ``fiftyone`` is not installed.
    """
    if not _FIFTYONE_AVAILABLE:
        raise ImportError("fiftyone is required for this function")
    sequence_results = {}
    for sequence_name in tqdm(view.distinct("sequence")):
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
    return sequence_results


def compute_and_save_sequence_metrics(
    csv_dirpath: str,
    view: fo.DatasetView,
    gt_field: str,
    pred_field: str,
    metric_fn: callable,
    *,
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

    Raises:
        ImportError: If ``fiftyone`` is not installed.
    """
    if not _FIFTYONE_AVAILABLE:
        raise ImportError("fiftyone is required for this function")
    base = name_separator.join(
        [view.dataset_name, gt_field, pred_field, metric_fn.__name__]
    )
    csv_name = (
        name_separator.join([base, csv_suffix] if csv_suffix else [base]) + ".csv"
    )
    csv_path = str(pathlib.Path(csv_dirpath) / csv_name)
    print(f"Saving metrics to {csv_path}")

    filtered_view = get_relevant_fields(view, [gt_field, pred_field, "sequence"])
    sequence_results = _collect_sequence_results(
        filtered_view, gt_field, pred_field, metric_fn, metric_kwargs, debug=debug
    )

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
    *,
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

    Raises:
        ImportError: If ``fiftyone`` is not installed.
    """
    if not _FIFTYONE_AVAILABLE:
        raise ImportError("fiftyone is required for this function")
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
    for sequence, (gt, pred) in sequence_results.items():
        try:
            metric.update(gt, pred, sequence)
        except (ValueError, IndexError) as e:
            metric.log_failed_sequence(sequence, gt, pred, exc=e)

    return metric


def _has_keyframes(seq_view: fo.DatasetView, pred_field: str) -> bool:
    """Return True if any frame in *seq_view* has keyframe data for *pred_field*.

    Args:
        seq_view: FiftyOne view for a single sequence.
        pred_field: Prediction field name to check.

    Returns:
        True if at least one keyframe value is truthy; False otherwise.

    Raises:
        ImportError: If ``fiftyone`` is not installed.
    """
    if not _FIFTYONE_AVAILABLE:
        raise ImportError("fiftyone is required for this function")
    video_view = (
        seq_view.select_group_slices(seq_view.default_group_slice)
        if seq_view.media_type == "group"
        else seq_view
    )
    try:
        kf_vals = video_view.values(f"frames[].{pred_field}.keyframe")
        return any(kf for kf in kf_vals if kf)
    except (ValueError, AttributeError, RuntimeError, TypeError, KeyError):
        return False


def _filter_valid_sequences(
    sequence_list: list,
    view: fo.DatasetView,
    pred_fields: list,
    instances: dict,
) -> list:
    """Validate keyframe availability and return sequences that have all fields.

    Sequences missing keyframes for any prediction field are logged as failed
    on every metric instance and excluded from the returned list.

    Args:
        sequence_list: Candidate sequence names.
        view: FiftyOne dataset view used to match individual sequences.
        pred_fields: Prediction field names to validate.
        instances: Nested dict ``{pred_field: {metric_name: metric_instance}}``.

    Returns:
        List of sequence names where all prediction fields have keyframe data.

    Raises:
        ImportError: If ``fiftyone`` is not installed.
    """
    if not _FIFTYONE_AVAILABLE:
        raise ImportError("fiftyone is required for this function")
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


def _run_metric_updates(
    valid_sequences: list,
    view: fo.DatasetView,
    pred_fields: list,
    gt_field: str,
    instances: dict,
) -> None:
    """Call ``update`` on every metric instance for each valid sequence.

    Args:
        valid_sequences: Sequence names confirmed to have keyframe data.
        view: FiftyOne dataset view used to match individual sequences.
        pred_fields: Prediction field names to evaluate.
        gt_field: FiftyOne field name for ground-truth detections.
        instances: Nested dict ``{pred_field: {metric_name: metric_instance}}``.

    Raises:
        ImportError: If ``fiftyone`` is not installed.
    """
    if not _FIFTYONE_AVAILABLE:
        raise ImportError("fiftyone is required for this function")
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


def compute_all_metrics_by_sequence(
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

    Raises:
        ImportError: If ``fiftyone`` is not installed.
        ValueError: If duplicate metric class names are found in *metrics*.

    Example:
        results = compute_all_metrics_by_sequence(
            view=view,
            gt_field="ground_truth_det",
            pred_fields=["model_a", "model_b"],
            metrics=[(TrackingMetrics, {"max_iou": 0.5}), (HOTAMetrics, {})],
        )
        mot_df = results_to_df(results["model_a"]["TrackingMetrics"])
    """
    if not _FIFTYONE_AVAILABLE:
        raise ImportError("fiftyone is required for this function")
    if isinstance(pred_fields, str):
        pred_fields = [pred_fields]

    resolved: list = (
        list(
            get_relevant_fields(view, [gt_field, *pred_fields, "sequence"]).distinct(
                "sequence"
            )
        )
        if sequence_list is None
        else sequence_list
    )

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
    valid_sequences = _filter_valid_sequences(resolved, view, pred_fields, instances)
    _run_metric_updates(valid_sequences, view, pred_fields, gt_field, instances)
    return instances


def compute_sizes(view: fo.DatasetView, gt_field: str) -> list:
    """Compute bounding-box areas for all annotated objects in a sequence view.

    Args:
        view: FiftyOne dataset view for the sequence.
        gt_field: FiftyOne field name for ground-truth detections.

    Returns:
        List of ``[frame_idx, track_id, area]`` entries for every annotated object.

    Raises:
        ImportError: If ``fiftyone`` is not installed.
        ValueError: If the view contains no samples after field selection.
    """
    if not _FIFTYONE_AVAILABLE:
        raise ImportError("fiftyone is required for this function")
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

    Raises:
        ImportError: If ``fiftyone`` is not installed.
    """
    if not _FIFTYONE_AVAILABLE:
        raise ImportError("fiftyone is required for this function")
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
