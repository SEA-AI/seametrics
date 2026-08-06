"""Build a per-sequence detection comparison across one or more models.

Wraps the whole recipe: fetch a dataset in batches, evaluate every sequence for
every model, pool the comparable subset, and hand back something reportable.
See ``AGENTS.md`` in this package for the reasoning behind the defaults.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from seametrics.detection.utils import (
    OVERALL_KEY,
    aggregate_sequence_results,
    payload_to_det_metrics_by_sequence,
    payload_to_detection_verdicts,
    sequence_results_to_df,
    tag_detections,
)
from seametrics.payload import Payload
from seametrics.payload.processor import PayloadProcessor

#: Small enough that any non-zero overlap counts, large enough that a zero
#: overlap never does. See "IoU threshold" in AGENTS.md.
DEFAULT_IOU = 1e-9


@dataclass
class DetectionReport:
    """Per-sequence and pooled detection results for a set of models.

    Attributes:
        models: Model field names, in the order they were evaluated.
        per_sequence: ``{model: {sequence: results}}`` for every sequence the
            model could evaluate.
        overall: ``{model: results}`` pooled over `pooled_sequences` only, so
            every model covers the same sequences.
        pooled_sequences: Sequences every model evaluated — the comparable set.
        skipped: ``{model: [sequence, ...]}`` the model could not evaluate.
        keyframe_frames: ``{model: n}`` frames flagged as keyframes.
        total_frames: ``{model: n}`` frames seen.
        iou_threshold: The threshold the numbers were produced at.
        tagged: ``{model: {tag: n}}`` label tags written, empty unless
            *tag_predictions* was set.
    """

    models: List[str]
    per_sequence: Dict[str, Dict[str, dict]]
    overall: Dict[str, dict]
    pooled_sequences: List[str]
    skipped: Dict[str, List[str]]
    keyframe_frames: Dict[str, int] = field(default_factory=dict)
    total_frames: Dict[str, int] = field(default_factory=dict)
    iou_threshold: float = DEFAULT_IOU
    tagged: Dict[str, Dict[str, int]] = field(default_factory=dict)

    @property
    def excluded_sequences(self) -> List[str]:
        """Sequences left out of pooling because some model could not evaluate them."""
        return sorted({s for names in self.skipped.values() for s in names})

    def to_df(self) -> pd.DataFrame:
        """Return one row per (model, sequence, area range), with OVERALL rows.

        Returns:
            pd.DataFrame: Adds a ``model`` column and an ``in_overall`` flag
                marking which rows fed the pooled figure. Every evaluated
                sequence keeps a row, including ones excluded from pooling.
        """
        frames = []
        pooled = set(self.pooled_sequences)
        for model in self.models:
            rows = dict(self.per_sequence[model])
            if model in self.overall:
                rows[OVERALL_KEY] = self.overall[model]
            if not rows:
                continue
            df = sequence_results_to_df(rows)
            df.insert(0, "model", model)
            df.insert(1, "in_overall", [s in pooled for s in df["sequence"]])
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)


def _pool_comparable(
    per_sequence: Dict[str, Dict[str, dict]], models: List[str]
) -> Tuple[Dict[str, dict], List[str]]:
    """Pool each model over the sequences that every model evaluated.

    Pooling over each model's own sequences would compare different subsets, so
    the intersection is used instead.

    Args:
        per_sequence: ``{model: {sequence: results}}``.
        models: Models to pool.

    Returns:
        Tuple[Dict[str, dict], List[str]]: The pooled result per model, and the
            sorted list of sequences it was computed over.
    """
    if not models:
        return {}, []
    common = set(per_sequence.get(models[0], {}))
    for model in models[1:]:
        common &= set(per_sequence.get(model, {}))
    shared = sorted(common)
    if not shared:
        return {}, []
    overall = {
        model: aggregate_sequence_results(
            {seq: per_sequence[model][seq] for seq in shared}
        )
        for model in models
    }
    return overall, shared


def _usable_sequences(
    payload: Payload,
    model: str,
    keyframes_only: bool,
    skipped: Dict[str, List[str]],
    keyframe_frames: Dict[str, int],
    total_frames: Dict[str, int],
) -> Dict[str, object]:
    """Select the sequences a model can evaluate, recording the rest as skipped.

    Args:
        payload: The batch just fetched.
        model: Prediction field being evaluated.
        keyframes_only: Whether a missing keyframe mask disqualifies a sequence.
        skipped: Mutated with sequences this model cannot evaluate.
        keyframe_frames: Mutated with the running keyframe count.
        total_frames: Mutated with the running frame count.

    Returns:
        Dict[str, object]: ``{sequence_name: Sequence}`` for the usable ones.
    """
    usable = {}
    for name, sequence in payload.sequences.items():
        mask = getattr(sequence, "keyframes", {}).get(model)
        if keyframes_only and mask is None:
            skipped[model].append(name)
            continue
        usable[name] = sequence
        total_frames[model] += len(sequence[model])
        keyframe_frames[model] += sum(mask) if mask else 0
    return usable


def evaluate_models(
    dataset_name: str,
    gt_field: str,
    models: List[str],
    *,
    slices: Optional[List[str]] = None,
    sequence_list: Optional[List[str]] = None,
    keyframes_only: bool = False,
    iou_threshold: float = DEFAULT_IOU,
    batch_size: int = 8,
    label_mapping: Optional[Dict[str, int]] = None,
    class_agnostic: bool = True,
    tag_predictions: bool = False,
    progress: bool = True,
    **metric_kwargs: object,
) -> DetectionReport:
    """Evaluate several models per sequence and pool them comparably.

    Sequences are fetched a batch at a time and discarded once evaluated, so the
    whole dataset never sits in memory. Pooling at the end is exact because
    detection counts are additive across sequences.

    Args:
        dataset_name (str): FiftyOne dataset to evaluate.
        gt_field (str): Ground truth field name.
        models (List[str]): Prediction fields to compare.
        slices (List[str], optional): Group slices to use, e.g.
            ``["thermal_wide"]``. Defaults to the processor's automatic choice.
        sequence_list (List[str], optional): Restrict to these sequences.
        keyframes_only (bool, optional): Evaluate only the frames each model
            flagged as keyframes. Sequences with no keyframes for a model are
            skipped for that model and excluded from pooling for all of them.
            Defaults to False.
        iou_threshold (float, optional): Must be greater than zero — at zero a
            detection matches ground truth it does not overlap. Defaults to
            ``DEFAULT_IOU``.
        batch_size (int, optional): Sequences fetched per query. Defaults to 8.
        label_mapping (Dict[str, int], optional): For class-specific metrics.
        class_agnostic (bool, optional): Defaults to True.
        tag_predictions (bool, optional): **Writes to the dataset.** Tag every
            evaluated prediction ``TP`` or ``FP`` as a fiftyone label tag, while
            the batch is still loaded, so the tags always agree with the numbers
            reported here. Off by default because it mutates a shared dataset and
            roughly quadruples the runtime — the cost is fiftyone saving frames,
            not the metric. Defaults to False.
        progress (bool, optional): Print batch progress. Defaults to True.
        **metric_kwargs: Forwarded to ``PrecisionRecallF1Support``, e.g.
            ``area_ranges`` and ``area_ranges_labels``.

    Returns:
        DetectionReport: Per-sequence results, pooled totals and coverage. When
            *tag_predictions* is set, ``report.tagged`` records how many tags were
            written per model.

    Raises:
        ValueError: If *models* is empty or *iou_threshold* is not positive.
    """
    if not models:
        raise ValueError("Provide at least one model field.")
    if iou_threshold <= 0:
        raise ValueError(
            f"iou_threshold must be > 0, got {iou_threshold}. At zero a detection"
            " matches ground truth it does not overlap at all."
        )

    import fiftyone as fo  # local: fiftyone is an optional extra
    from fiftyone import ViewField as F

    view = fo.load_dataset(dataset_name)
    if slices and view.media_type == "group":
        view = view.select_group_slices(slices)
    sequences = sorted(sequence_list or view.distinct("sequence"))

    per_sequence: Dict[str, Dict[str, dict]] = {m: {} for m in models}
    skipped: Dict[str, List[str]] = {m: [] for m in models}
    keyframe_frames = dict.fromkeys(models, 0)
    total_frames = dict.fromkeys(models, 0)
    tagged: Dict[str, Dict[str, int]] = {m: {} for m in models}

    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        payload = PayloadProcessor(
            dataset_name=dataset_name,
            gt_field=gt_field,
            models=models,
            slices=slices,
            sequence_list=batch,
            batch_size=batch_size,
        ).payload

        for model in models:
            usable = _usable_sequences(
                payload, model, keyframes_only, skipped, keyframe_frames, total_frames
            )
            if not usable:
                continue
            usable_payload = Payload(
                dataset=dataset_name,
                models=models,
                gt_field_name=gt_field,
                sequences=usable,
            )
            shared = dict(
                model_name=model,
                label_mapping=label_mapping,
                class_agnostic=class_agnostic,
                keyframes_only=keyframes_only,
                iou_thresholds=[iou_threshold],
                **metric_kwargs,
            )
            per_sequence[model].update(
                payload_to_det_metrics_by_sequence(
                    usable_payload, include_overall=False, **shared
                )
            )
            if tag_predictions:
                # re-running the metric is cheap next to fetching the batch, and
                # reusing the tested path beats duplicating the extraction here
                written = tag_detections(
                    view.match(F("sequence").is_in(list(usable))),
                    model,
                    payload_to_detection_verdicts(usable_payload, **shared),
                    progress=progress,
                )
                for tag, count in written.items():
                    tagged[model][tag] = tagged[model].get(tag, 0) + count

        del payload
        if progress:
            done = min(start + batch_size, len(sequences))
            print(f"  {done}/{len(sequences)} sequences", flush=True)

    overall, pooled = _pool_comparable(per_sequence, models)
    return DetectionReport(
        models=list(models),
        per_sequence=per_sequence,
        overall=overall,
        pooled_sequences=pooled,
        skipped=skipped,
        keyframe_frames=keyframe_frames,
        total_frames=total_frames,
        iou_threshold=iou_threshold,
        tagged=tagged if tag_predictions else {},
    )
