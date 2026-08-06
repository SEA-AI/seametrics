import contextlib
import io
import os
from typing import Dict, Iterator, List, Optional, Tuple, Union

import fiftyone as fo
import numpy as np
import pandas as pd
from deprecated import deprecated
from fiftyone import ViewField as F
from tqdm import tqdm

from seametrics.detection.imports import _TORCHMETRICS_AVAILABLE
from seametrics.detection.np.utils import box_convert
from seametrics.payload import Payload, Sequence

if _TORCHMETRICS_AVAILABLE:
    from torch import tensor

# payload functions

# Key under which the pooled result is stored alongside the per-sequence results.
OVERALL_KEY = "OVERALL"

# Count fields that are additive across sequences. Ratios (precision/recall/f1)
# are NOT additive and must be recomputed from the pooled counts.
_ADDITIVE_KEYS = ("tp", "fp", "fn", "duplicates", "fpi", "nImgs")


def _sequence_keyframes(
    sequence: Sequence, sequence_name: str, model_name: str
) -> List[bool]:
    """Return the keyframe mask a sequence recorded for one prediction field.

    Args:
        sequence (Sequence): The payload sequence.
        sequence_name (str): Name of the sequence, used in error messages.
        model_name (str): Prediction field whose mask is wanted.

    Returns:
        List[bool]: One flag per frame.

    Raises:
        ValueError: If the sequence carries no keyframe mask for *model_name*.
            Filtering silently on a missing mask would evaluate every frame while
            reporting keyframe-only numbers.
    """
    keyframes = getattr(sequence, "keyframes", None) or {}
    mask = keyframes.get(model_name)
    if mask is None:
        raise ValueError(
            f"Sequence {sequence_name!r} has no keyframe data for {model_name!r}."
            " Rebuild the payload with a PayloadProcessor that records keyframes,"
            " or pass keyframes_only=False."
        )
    return mask


def _filter_to_keyframes(
    frames: List[List[fo.Detection]],
    mask: List[bool],
    sequence_name: str,
    field_name: str,
) -> List[List[fo.Detection]]:
    """Drop the frames whose keyframe flag is falsy.

    Applied to ground truth and predictions alike, mirroring
    ``seametrics.tracking.utils.build_detection_inputs``: a frame the model never
    emitted on is removed from the evaluation entirely rather than counted as a
    miss, and stops contributing to ``nImgs``.

    Args:
        frames (List[List[fo.Detection]]): Per-frame detection lists.
        mask (List[bool]): One keyframe flag per frame.
        sequence_name (str): Name of the sequence, used in error messages.
        field_name (str): Field being filtered, used in error messages.

    Returns:
        List[List[fo.Detection]]: Only the frames flagged as keyframes.

    Raises:
        ValueError: If *mask* and *frames* have different lengths, which would
            silently misalign ground truth and predictions.
    """
    if len(mask) != len(frames):
        raise ValueError(
            f"Keyframe mask for sequence {sequence_name!r} has {len(mask)} entries"
            f" but field {field_name!r} has {len(frames)} frames."
        )
    return [
        frame for frame, is_keyframe in zip(frames, mask, strict=True) if is_keyframe
    ]


def payload_to_det_metric(
    payload: Payload,
    model_name: str = None,
    label_mapping: Dict[str, int] = None,
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
            keyframes, dropping ground truth and predictions alike, as
            ``seametrics.tracking`` does. Frames the model never emitted on are
            excluded from the evaluation instead of counting as misses. Requires a
            payload built by a ``PayloadProcessor`` that records keyframes.
            Defaults to False.

    Returns:
        Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]]]:
            A tuple containing the converted (predictions, references). Boxes are
            in absolute ``xywh`` pixel coordinates, so the metric consuming them
            must be created with ``box_format="xywh"``.
    """
    if class_agnostic and label_mapping is not None:
        raise ValueError("Label mapping cannot be provided for class-agnostic metrics.")

    predictions, references = [], []

    if model_name is None:
        model_name = payload.models[0]

    for sequence_name, sequence in payload.sequences.items():
        w, h = (
            sequence.resolution.width,
            sequence.resolution.height,
        )
        pred_frames = sequence[model_name]
        gt_frames = sequence[payload.gt_field_name]

        if keyframes_only:
            mask = _sequence_keyframes(sequence, sequence_name, model_name)
            pred_frames = _filter_to_keyframes(
                pred_frames, mask, sequence_name, model_name
            )
            gt_frames = _filter_to_keyframes(
                gt_frames, mask, sequence_name, payload.gt_field_name
            )

        predictions.extend(
            payload_sequence_to_det_metrics(
                sequence_dets=pred_frames, w=w, h=h, label_mapping=label_mapping
            )
        )
        references.extend(
            payload_sequence_to_det_metrics(
                sequence_dets=gt_frames,
                w=w,
                h=h,
                is_gt=True,
                label_mapping=label_mapping,
            )
        )

    return predictions, references


def payload_to_det_metrics_by_sequence(
    payload: Payload,
    model_name: Optional[str] = None,
    label_mapping: Optional[Dict[str, int]] = None,
    class_agnostic: bool = True,
    include_overall: bool = True,
    keyframes_only: bool = False,
    **metric_kwargs: object,
) -> Dict[str, dict]:
    """Evaluate every sequence of a payload separately, plus a pooled total.

    Unlike :func:`payload_to_det_metric`, which flattens all sequences into one
    pooled evaluation, this runs an independent ``PrecisionRecallF1Support`` per
    sequence. Use it to see which sequences regressed: the pooled precision and
    recall are support-weighted, so a single long sequence can hide a regression
    in a short one.

    Boxes are converted with :func:`frame_dets_to_det_metrics`, which emits
    absolute ``xywh``, so ``box_format`` is fixed to ``"xywh"`` and may not be
    overridden.

    Args:
        payload (Payload): The payload containing sequences, models and the
            ground truth field name.
        model_name (str, optional): The name of the model to evaluate. If not
            provided, the first model in the payload is used.
        label_mapping (Dict[str, int], optional): Dictionary mapping string labels
            to numbers, required for class-specific metrics. Defaults to None.
        class_agnostic (bool, optional): Flag indicating if the metrics should be
            calculated in a class-agnostic way. Defaults to True.
        include_overall (bool, optional): Add an ``"OVERALL"`` entry holding the
            pooled result. Computed by summing the per-sequence counts, so the
            data is still traversed exactly once. Defaults to True.
        keyframes_only (bool, optional): Keep only the frames *model_name* flagged
            as keyframes, dropping ground truth and predictions alike, as
            ``seametrics.tracking`` does. ``nImgs`` then counts keyframes only.
            Requires a payload built by a ``PayloadProcessor`` that records
            keyframes. Defaults to False.
        **metric_kwargs: Forwarded to ``PrecisionRecallF1Support`` (e.g.
            ``iou_thresholds``, ``area_ranges``, ``area_ranges_labels``). When
            *label_mapping* is given and ``labels`` is not, ``labels`` defaults to
            the sorted mapping values so that every sequence reports the same
            classes in the same order — otherwise per-sequence result arrays would
            have different lengths and could not be compared.

    Returns:
        Dict[str, dict]: ``{sequence_name: results}``, where each value is the
            full ``PrecisionRecallF1Support.compute()`` output. When
            *include_overall* is set, a final ``"OVERALL"`` entry holds the pooled
            result and is byte-for-byte what :func:`payload_to_det_metric` plus a
            single metric would produce. Pass the whole dict straight to
            :func:`sequence_results_to_df` to get one row per
            (sequence, area range) with OVERALL last.

    Raises:
        ValueError: If *label_mapping* is combined with ``class_agnostic=True``,
            if ``box_format`` is passed in *metric_kwargs*, or if a sequence is
            literally named ``"OVERALL"`` while *include_overall* is set.

    Note:
        A sequence that fails to evaluate propagates the exception, aborting the
        whole call. This differs from ``seametrics.tracking``, where per-sequence
        errors are routed to ``metric.failed_sequences``.

    Note:
        Keyframe masks are per prediction field, so two models with different
        keyframes are evaluated over different frame subsets. Their numbers are
        each internally consistent but not strictly comparable to one another.
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
    for sequence_name, sequence in payload.sequences.items():
        w, h = sequence.resolution.width, sequence.resolution.height

        pred_frames = sequence[model_name]
        gt_frames = sequence[payload.gt_field_name]

        if keyframes_only:
            mask = _sequence_keyframes(sequence, sequence_name, model_name)
            pred_frames = _filter_to_keyframes(
                pred_frames, mask, sequence_name, model_name
            )
            gt_frames = _filter_to_keyframes(
                gt_frames, mask, sequence_name, payload.gt_field_name
            )

        predictions = payload_sequence_to_det_metrics(
            sequence_dets=pred_frames,
            w=w,
            h=h,
            label_mapping=label_mapping,
        )
        references = payload_sequence_to_det_metrics(
            sequence_dets=gt_frames,
            w=w,
            h=h,
            is_gt=True,
            label_mapping=label_mapping,
        )

        metric = PrecisionRecallF1Support(
            box_format="xywh",
            class_agnostic=class_agnostic,
            **metric_kwargs,
        )
        metric.update(predictions, references)
        sequence_results[sequence_name] = metric.compute()

    if include_overall and sequence_results:
        sequence_results[OVERALL_KEY] = aggregate_sequence_results(sequence_results)

    return sequence_results


def _pool_area_range_metrics(entries: List[dict]) -> dict:
    """Pool one area range's per-sequence metric dicts into a single dict.

    Counts are summed; precision, recall and f1 are recomputed from those sums
    using the same ``-1`` sentinel rules as ``COCOeval._summarize_pr_rec_f1``, so
    the result is indistinguishable from evaluating every frame in one metric.

    Args:
        entries (List[dict]): One metric dict per sequence, all for the same area
            range and produced with the same metric configuration.

    Returns:
        dict: A pooled metric dict with the same keys as its inputs.

    Raises:
        ValueError: If a count field has inconsistent shapes across sequences,
            which happens in class-specific mode when the sequences were not
            evaluated against a shared ``labels`` list.
    """
    totals = {}
    for key in _ADDITIVE_KEYS:
        values = [np.asarray(entry[key]) for entry in entries]
        shapes = {value.shape for value in values}
        if len(shapes) > 1:
            raise ValueError(
                f"Inconsistent `{key}` shapes across sequences ({sorted(shapes)})."
                " In class-specific mode pass a shared `labels` list so every"
                " sequence reports the same classes."
            )
        totals[key] = np.sum(values, axis=0)

    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    support = tp + fn

    # mirror COCOeval: compute, then overwrite undefined entries with -1
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp == 0, -1.0, tp / (tp + fp))
        recall = np.where(tp + fn == 0, -1.0, tp / (tp + fn))
        f1 = np.where(
            (precision == -1) | (recall == -1) | (precision + recall == 0),
            -1.0,
            2 * precision * recall / (precision + recall),
        )

    def _as_scalar_or_array(
        value: np.ndarray, cast: type
    ) -> Union[int, float, np.ndarray]:
        """Return a python scalar for 0-d input, else the array, like COCOeval."""
        return cast(value) if value.ndim == 0 else value

    first = entries[0]
    return {
        "range": first["range"],
        "iouThr": first["iouThr"],
        "maxDets": first["maxDets"],
        "tp": _as_scalar_or_array(tp, int),
        "fp": _as_scalar_or_array(fp, int),
        "fn": _as_scalar_or_array(fn, int),
        "duplicates": _as_scalar_or_array(totals["duplicates"], int),
        "precision": _as_scalar_or_array(precision, float),
        "recall": _as_scalar_or_array(recall, float),
        "f1": _as_scalar_or_array(f1, float),
        "support": _as_scalar_or_array(support, int),
        "fpi": _as_scalar_or_array(totals["fpi"], int),
        "nImgs": int(np.sum([entry["nImgs"] for entry in entries])),
    }


def aggregate_sequence_results(sequence_results: Dict[str, dict]) -> dict:
    """Pool per-sequence detection results into one overall result.

    Detection counts are additive across sequences because matching never crosses
    a frame boundary, so the pooled numbers can be obtained by summing tp, fp, fn,
    duplicates and fpi and recomputing the ratios once. No second pass over the
    detections is needed.

    Each area range is pooled independently, mirroring ``COCOeval.summarize``.
    Note that area ranges are not nested pools: "small" + "large" does not sum to
    "all", because ground truth outside a range is ignored rather than counted.

    Args:
        sequence_results (Dict[str, dict]): ``{sequence_name: results}`` as
            returned by :func:`payload_to_det_metrics_by_sequence`. Any existing
            ``"OVERALL"`` entry is skipped so the function is idempotent.

    Returns:
        dict: A results dict shaped like ``PrecisionRecallF1Support.compute()``,
            containing only the ``"metrics"`` key (the per-sequence ``params`` and
            ``eval`` objects cannot be meaningfully pooled).

    Raises:
        ValueError: If *sequence_results* holds no poolable sequences, if the
            sequences disagree on their area-range labels, or if a count field has
            inconsistent shapes across sequences.
    """
    poolable = {
        name: results
        for name, results in sequence_results.items()
        if name != OVERALL_KEY
    }
    if not poolable:
        raise ValueError("No sequence results to pool.")

    area_label_sets = {tuple(results["metrics"]) for results in poolable.values()}
    if len(area_label_sets) > 1:
        raise ValueError(
            f"Sequences disagree on area-range labels: {sorted(area_label_sets)}."
            " Pool only results produced with the same metric configuration."
        )

    area_labels = next(iter(area_label_sets))
    return {
        "metrics": {
            area_label: _pool_area_range_metrics(
                [results["metrics"][area_label] for results in poolable.values()]
            )
            for area_label in area_labels
        }
    }


def payload_to_detection_verdicts(
    payload: Payload,
    model_name: Optional[str] = None,
    label_mapping: Optional[Dict[str, int]] = None,
    class_agnostic: bool = True,
    keyframes_only: bool = False,
    area_range_label: str = "all",
    **metric_kwargs: object,
) -> Dict[str, Dict[str, str]]:
    """Classify every prediction in a payload as TP, FP or ignored.

    Runs the same evaluation as :func:`payload_to_det_metrics_by_sequence` but
    returns the per-detection outcome instead of the aggregates, keyed by the
    fiftyone detection id so the result can be written back to a dataset without
    any frame arithmetic. Feed it to :func:`tag_detections`.

    Args:
        payload (Payload): The payload containing sequences, models and the
            ground truth field name.
        model_name (str, optional): Model to classify. Defaults to the first.
        label_mapping (Dict[str, int], optional): As for
            :func:`payload_to_det_metrics_by_sequence`.
        class_agnostic (bool, optional): As for
            :func:`payload_to_det_metrics_by_sequence`. Defaults to True.
        keyframes_only (bool, optional): Classify only the frames the model
            flagged as keyframes. Detections on other frames get no verdict at
            all, because they were never evaluated. Defaults to False.
        area_range_label (str, optional): Which area range the verdicts describe.
            A detection outside the range is ignored rather than scored, so the
            verdict is range-specific. Defaults to "all".
        **metric_kwargs: Forwarded to ``PrecisionRecallF1Support``.

    Returns:
        Dict[str, Dict[str, str]]: ``{sequence_name: {detection_id: verdict}}``
            where verdict is ``"TP"``, ``"FP"`` or ``"ignored"``.

    Raises:
        ValueError: If *label_mapping* is combined with ``class_agnostic=True``,
            or if ``box_format`` is passed in *metric_kwargs*.
    """
    from seametrics.detection import PrecisionRecallF1Support

    if class_agnostic and label_mapping is not None:
        raise ValueError("Label mapping cannot be provided for class-agnostic metrics.")
    if "box_format" in metric_kwargs:
        raise ValueError(
            "`box_format` cannot be overridden: frame_dets_to_det_metrics always"
            " produces absolute xywh boxes."
        )
    if model_name is None:
        model_name = payload.models[0]
    if label_mapping is not None:
        metric_kwargs.setdefault("labels", sorted(set(label_mapping.values())))

    verdicts = {}
    for sequence_name, sequence in payload.sequences.items():
        w, h = sequence.resolution.width, sequence.resolution.height
        pred_frames = sequence[model_name]
        gt_frames = sequence[payload.gt_field_name]

        if keyframes_only:
            mask = _sequence_keyframes(sequence, sequence_name, model_name)
            pred_frames = _filter_to_keyframes(
                pred_frames, mask, sequence_name, model_name
            )
            gt_frames = _filter_to_keyframes(
                gt_frames, mask, sequence_name, payload.gt_field_name
            )

        # the converter drops detections whose label is unmapped, so rebuild the
        # surviving id order exactly the way it builds the boxes
        ids_by_frame = [
            [
                det.id
                for det in frame
                if not (label_mapping and det["label"] not in label_mapping)
            ]
            for frame in pred_frames
        ]

        metric = PrecisionRecallF1Support(
            box_format="xywh", class_agnostic=class_agnostic, **metric_kwargs
        )
        metric.update(
            payload_sequence_to_det_metrics(
                sequence_dets=pred_frames, w=w, h=h, label_mapping=label_mapping
            ),
            payload_sequence_to_det_metrics(
                sequence_dets=gt_frames,
                w=w,
                h=h,
                is_gt=True,
                label_mapping=label_mapping,
            ),
        )
        with contextlib.redirect_stdout(io.StringIO()):
            metric.compute()

        per_sequence = {}
        for (frame_index, det_index), verdict in metric.detection_verdicts(
            area_range_label=area_range_label
        ).items():
            per_sequence[ids_by_frame[frame_index][det_index]] = verdict
        verdicts[sequence_name] = per_sequence

    return verdicts


def tag_detections(
    dataset: object,
    field_name: str,
    verdicts: Dict[str, Dict[str, str]],
    tags: Tuple[str, ...] = ("TP", "FP"),
    clear_first: bool = True,
) -> Dict[str, int]:
    """Write TP/FP verdicts onto detections as fiftyone label tags.

    Label tags are what the app filters on, so tagging makes the outcome
    browsable. This mutates the dataset.

    Args:
        dataset (fo.Dataset | fo.DatasetView): Dataset or view holding *field_name*.
        field_name (str): Prediction field whose detections to tag.
        verdicts (Dict[str, Dict[str, str]]): As returned by
            :func:`payload_to_detection_verdicts`.
        tags (Tuple[str, ...], optional): Which verdicts to write. Defaults to
            ``("TP", "FP")``; add ``"ignored"`` to tag those too.
        clear_first (bool, optional): Remove any of *tags* already present on the
            field's detections before writing, so re-running does not accumulate
            stale tags. Defaults to True.

    Returns:
        Dict[str, int]: How many detections received each tag, plus ``"cleared"``.
    """
    wanted = set(tags)
    lookup = {}
    for per_sequence in verdicts.values():
        for detection_id, verdict in per_sequence.items():
            if verdict in wanted:
                lookup[detection_id] = verdict

    counts = dict.fromkeys(wanted, 0)
    counts["cleared"] = 0
    is_video = dataset.media_type == "video"

    def apply(detection: object) -> None:
        """Retag one detection in place, recording what changed."""
        if clear_first and detection.tags:
            kept = [t for t in detection.tags if t not in wanted]
            counts["cleared"] += len(detection.tags) - len(kept)
            detection.tags = kept
        verdict = lookup.get(detection.id)
        if verdict is not None:
            detection.tags = [*detection.tags, verdict]
            counts[verdict] += 1

    for sample in dataset.iter_samples(autosave=True, progress=True):
        containers = list(sample.frames.values()) if is_video else [sample]
        for detection in _iter_detections(containers, field_name):
            apply(detection)
    return counts


def _iter_detections(containers: list, field_name: str) -> Iterator[fo.Detection]:
    """Yield every detection of *field_name* across samples or video frames.

    Args:
        containers (list): Samples, or the frames of one video sample.
        field_name (str): Detections field to read.

    Yields:
        fo.Detection: Each detection present on the field.
    """
    for container in containers:
        detections = container[field_name]
        if detections is not None:
            yield from detections.detections


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
            one dict per frame. Boxes are absolute ``xywh`` — see
            :func:`frame_dets_to_det_metrics`.
    """
    output = []

    for frame_dets in sequence_dets:
        frame_dict = frame_dets_to_det_metrics(frame_dets, w, h, is_gt, label_mapping)
        output.append(frame_dict)

    return output


def _denormalize_fo_bbox(bounding_box: List[float], w: int, h: int) -> List[float]:
    """Scale a relative fiftyone bounding box to absolute pixel coordinates.

    Args:
        bounding_box (List[float]): Relative ``[x, y, width, height]`` as stored
            by fiftyone, with ``(x, y)`` the top-left corner.
        w (int): Width in pixels of the image.
        h (int): Height in pixels of the image.

    Returns:
        List[float]: ``[x, y, width, height]`` in absolute pixel coordinates.
    """
    rel_x, rel_y, rel_w, rel_h = bounding_box
    return [rel_x * w, rel_y * h, rel_w * w, rel_h * h]


def frame_dets_to_det_metrics(
    fo_dets: List[fo.Detection],
    w: int,
    h: int,
    is_gt: bool = False,
    label_mapping: Dict[str, int] = None,
) -> Dict[str, np.ndarray]:
    """Convert a list of fiftyone detections to format of PrecisionRecallF1.

    FiftyOne stores ``Detection.bounding_box`` as ``[x, y, width, height]``
    *relative* to the image size, with ``(x, y)`` the top-left corner. The boxes
    returned here keep that ``xywh`` layout but are scaled to absolute pixel
    coordinates, so the consuming metric must be constructed with
    ``box_format="xywh"``.

    Args:
        fo_dets (List[fo.Detection]): A list of fiftyone detections belonging to
            a single frame.
        w (int): Width in pixels of the image.
        h (int): Height in pixels of the image.
        is_gt (bool, optional): Flag indicating if the input data is ground truth.
            Defaults to False.
        label_mapping (Dict[str, int], optional): Dictionary mapping string labels to
            numbers, which should be provided if the detection metrics should be
            calculated in a class-specific way. Detections whose label is missing
            from the mapping are dropped. Defaults to None.

    Returns:
        Dict[str, np.ndarray]: ``boxes`` of shape ``(N, 4)`` in absolute ``xywh``
            pixel coordinates and ``labels`` of shape ``(N,)``. Predictions
            additionally carry ``scores`` (``1.0`` when a detection has no
            confidence). Empty input yields empty (1-D) arrays.

            No ``area`` is emitted: ``PrecisionRecallF1Support`` always derives
            the area from the bounding box, so a fiftyone ``area`` attribute
            would be silently discarded. Area-range bucketing therefore uses the
            same geometry as the IoU.
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

        boxes.append(_denormalize_fo_bbox(det["bounding_box"], w, h))
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
def prepare_data_for_det_metrics(
    gt_bboxes_per_frame,
    gt_labels_per_frame,
    dt_bboxes_per_frame,
    dt_labels_per_frame,
    dt_scores_per_frame,
    img_w,
    img_h,
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
    if not os.path.exists(csv_dirpath):
        os.makedirs(csv_dirpath)
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
    """Denormalize boxes from [0, 1] to [0, img_w] and [0, img_h].

    The box layout only matters insofar as x-like values must sit at even column
    indices and y-like values at odd ones, which holds for ``xyxy``, ``xywh`` and
    ``cxcywh``. Boxes containing any value greater than ``1.0`` are assumed to be
    in pixel coordinates already and are returned untouched.

    Args:
        boxes (np.ndarray): Boxes which will be denormalized, shape ``(N, 4)``.
        img_w (int): Width of image in pixels.
        img_h (int): Height of image in pixels.

    Returns:
        np.ndarray: Denormalized boxes of shape ``(N, 4)`` and floating-point
            dtype. The input array is never modified in place.
    """
    if boxes.size == 0:
        return boxes

    # check if boxes are normalized
    if np.any(boxes > 1.0):
        return boxes

    # copy so callers keep their normalized boxes, and cast so that integer
    # inputs (e.g. an all-zero/one box) do not truncate the scaled values
    boxes = boxes.astype(np.float64)
    boxes[:, 0::2] *= img_w
    boxes[:, 1::2] *= img_h
    return boxes
