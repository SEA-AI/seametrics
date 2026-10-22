"""Build a per-sequence detection comparison across one or more models.

Wraps the whole recipe: fetch a dataset in batches, evaluate every sequence for
every model, pool the comparable subset, and hand back something reportable.
See ``AGENTS.md`` in this package for the reasoning behind the defaults.
"""

import dataclasses
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import fiftyone as fo
import numpy as np
import pandas as pd
from fiftyone import ViewField as F
from tqdm import tqdm

from seametrics.constants import EXCLUDED_CLASSES
from seametrics.detection.pooling import OVERALL_KEY, aggregate_sequence_results
from seametrics.detection.utils import (
    payload_to_det_metrics_by_sequence,
    sequence_results_to_df,
)
from seametrics.payload import Payload
from seametrics.payload.processor import PayloadProcessor

#: Small enough that any pixel-aligned overlap counts, large enough that a zero
#: overlap never does. See "IoU threshold" in AGENTS.md.
DEFAULT_IOU = 1e-9


@dataclass(frozen=True)
class EvaluationOptions:
    """How every model is evaluated; the metrics and FiftyOne's evaluation share it.

    Attributes:
        keyframes_only: Evaluate only the frames each model flagged as
            keyframes. Sequences with no keyframes for a model are skipped for
            that model and excluded from pooling for all of them.
        iou_threshold: Must be positive: at zero a detection matches ground
            truth it does not overlap.
        label_mapping: For class-specific metrics.
        class_agnostic: Pool every class together.

    Raises:
        ValueError: If *iou_threshold* is not positive.
    """

    keyframes_only: bool = False
    iou_threshold: float = DEFAULT_IOU
    label_mapping: Optional[Dict[str, int]] = None
    class_agnostic: bool = True

    def __post_init__(self) -> None:
        """Reject a threshold at which non-overlapping boxes would match."""
        if self.iou_threshold <= 0:
            raise ValueError(
                f"iou_threshold must be > 0, got {self.iou_threshold}. At zero a"
                " detection matches ground truth it does not overlap at all."
            )

    def evaluation_kwargs(self, model: str, metric_kwargs: Dict[str, object]) -> dict:
        """Keyword arguments for the per-sequence metrics of *model*.

        Raises:
            ValueError: If *metric_kwargs* sets a key these options own.
        """
        options = {
            "model_name": model,
            "label_mapping": self.label_mapping,
            "class_agnostic": self.class_agnostic,
            "keyframes_only": self.keyframes_only,
            "iou_thresholds": [self.iou_threshold],
        }
        if overridden := sorted(options.keys() & metric_kwargs.keys()):
            raise ValueError(f"Set {overridden} through EvaluationOptions instead.")
        return {**options, **metric_kwargs}


@dataclass
class DetectionReport:
    """Per-sequence and pooled detection results for a set of models.

    Attributes:
        models: Model field names, in the order they were evaluated.
        per_sequence: ``{model: {sequence: results}}`` for every sequence the
            model could evaluate.
        overall: ``{model: results}`` pooled over `pooled_sequences` only, so
            every model covers the same sequences.
        pooled_sequences: Sequences every model evaluated: the comparable set.
        skipped: ``{model: [sequence, ...]}`` the model could not evaluate.
    """

    models: List[str]
    per_sequence: Dict[str, Dict[str, dict]]
    overall: Dict[str, dict]
    pooled_sequences: List[str]
    skipped: Dict[str, List[str]]

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
    shared = sorted(set.intersection(*(set(per_sequence[m]) for m in models)))
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
    report: DetectionReport, payload: Payload, model: str, keyframes_only: bool
) -> Dict[str, object]:
    """Select the sequences a model can evaluate, recording the rest as skipped.

    Args:
        report: Mutated: skipped sequences.
        payload: The batch just fetched.
        model: Prediction field being evaluated.
        keyframes_only: Whether a missing keyframe mask disqualifies a sequence.

    Returns:
        Dict[str, object]: ``{sequence_name: Sequence}`` for the usable ones.
    """
    usable = {}
    for name, sequence in payload.sequences.items():
        mask = (getattr(sequence, "keyframes", None) or {}).get(model)
        if keyframes_only and mask is None:
            report.skipped[model].append(name)
            continue
        usable[name] = sequence
    return usable


def _evaluate_batch(
    report: DetectionReport,
    payload: Payload,
    *,
    options: EvaluationOptions,
    metric_kwargs: Dict[str, object],
) -> None:
    """Evaluate every model on one fetched batch, adding to *report* in place.

    Args:
        report: Mutated: per-sequence results and skips.
        payload: The batch just fetched.
        options: How to evaluate.
        metric_kwargs: Forwarded to ``PrecisionRecallF1Support``.
    """
    for model in report.models:
        usable = _usable_sequences(report, payload, model, options.keyframes_only)
        if not usable:
            continue
        report.per_sequence[model].update(
            payload_to_det_metrics_by_sequence(
                dataclasses.replace(payload, sequences=usable),
                **options.evaluation_kwargs(model, metric_kwargs),
            )
        )


def _write_evaluations(
    view: fo.DatasetView,
    gt_field: str,
    report: DetectionReport,
    options: EvaluationOptions,
    eval_key: str,
) -> None:
    """Run FiftyOne's evaluation on the frames and labels *report* evaluated.

    FiftyOne's COCO matching marks every box tp, fp or fn under
    ``<eval_key>_<model>``, browsable in the app. Its counts must equal the
    ``"all"`` area range reported here; a mismatch is warned about.

    Args:
        view: The dataset view the report was computed on.
        gt_field: Ground truth field name.
        report: The report to check FiftyOne's counts against.
        options: The options the report was computed with.
        eval_key: Prefix of the evaluation key of each model.
    """
    prefix = "frames." if view.media_type == "video" else ""
    labels = ~F("label").is_in(EXCLUDED_CLASSES)
    if options.label_mapping:
        labels &= F("label").is_in(list(options.label_mapping))
    for model in report.models:
        key = f"{eval_key}_{model}"
        model_view = view.match(F("sequence").is_in(sorted(report.per_sequence[model])))
        if options.keyframes_only:
            keyframe = F(f"{model}.keyframe").is_in([True])
            model_view = (
                model_view.match_frames(keyframe)
                if prefix
                else model_view.match(keyframe)
            )
        for name in (gt_field, model):
            model_view = model_view.filter_labels(
                prefix + name, labels, only_matches=False
            )
            if options.class_agnostic:
                # classwise=False still scores a cross-class match fp/fn; one
                # label in the view (not the dataset) makes it a tp, as ours
                model_view = model_view.set_field(
                    f"{prefix}{name}.detections.label", "object"
                )
        if key in view.list_evaluations():
            view.delete_evaluation(key)  # a rerun replaces, never accumulates
        model_view.evaluate_detections(
            prefix + model,
            gt_field=prefix + gt_field,
            eval_key=key,
            iou=options.iou_threshold,
            classwise=not options.class_agnostic,
            progress=False,
        )
        _warn_on_disagreement(model_view, key, report.per_sequence[model])


def _warn_on_disagreement(
    view: fo.DatasetView, key: str, per_sequence: Dict[str, dict]
) -> None:
    """Warn if FiftyOne's tp/fp under *key* differ from the ``"all"`` results."""
    ours = [results["metrics"].get("all") for results in per_sequence.values()]
    if not ours or not all(ours):
        return
    expected = (
        sum(np.sum(m["tp"]) for m in ours),
        sum(np.sum(m["fp"]) for m in ours),
    )
    found = (view.sum(f"{key}_tp"), view.sum(f"{key}_fp"))
    if found != expected:
        warnings.warn(
            f"FiftyOne evaluation {key!r} counts tp, fp = {found}, but the report"
            f" has {expected}.",
            stacklevel=3,
        )


def evaluate_models(
    dataset_name: str,
    gt_field: str,
    models: List[str],
    *,
    options: Optional[EvaluationOptions] = None,
    slices: Optional[List[str]] = None,
    sequence_list: Optional[List[str]] = None,
    batch_size: int = 8,
    eval_key: Optional[str] = None,
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
        options (EvaluationOptions, optional): Keyframes, IoU threshold and
            classes. Defaults to ``EvaluationOptions()``.
        slices (List[str], optional): Group slices to use, e.g.
            ``["thermal_wide"]``. Defaults to the processor's automatic choice.
        sequence_list (List[str], optional): Restrict to these sequences.
        batch_size (int, optional): Sequences fetched per query. Defaults to 8.
        eval_key (str, optional): **Writes to the dataset.** Also run FiftyOne's
            evaluation as ``<eval_key>_<model>`` on the same frames, labels and
            IoU, marking every box tp, fp or fn. Defaults to None.
        **metric_kwargs: Forwarded to ``PrecisionRecallF1Support``, e.g.
            ``area_ranges`` and ``area_ranges_labels``.

    Returns:
        DetectionReport: Per-sequence results, pooled totals and coverage.

    Raises:
        ValueError: If *models* is empty.
    """
    if not models:
        raise ValueError("Provide at least one model field.")
    options = options or EvaluationOptions()

    view = fo.load_dataset(dataset_name)
    if slices and view.media_type == "group":
        view = view.select_group_slices(slices)
    if eval_key and view.media_type == "group":
        raise ValueError("Evaluating a group dataset needs `slices`, the ones used.")
    if sequence_list is None:
        sequence_list = view.distinct("sequence")
    sequences = sorted(sequence_list)

    report = DetectionReport(
        models=list(models),
        per_sequence={m: {} for m in models},
        overall={},
        pooled_sequences=[],
        skipped={m: [] for m in models},
    )
    for start in tqdm(range(0, len(sequences), batch_size), desc="Batches"):
        # the batch payload is dropped once evaluated, so only one is in memory
        _evaluate_batch(
            report,
            PayloadProcessor(
                dataset_name=dataset_name,
                gt_field=gt_field,
                models=models,
                slices=slices,
                sequence_list=sequences[start : start + batch_size],
                batch_size=batch_size,
            ).payload,
            options=options,
            metric_kwargs=metric_kwargs,
        )

    report.overall, report.pooled_sequences = _pool_comparable(
        report.per_sequence, models
    )
    if eval_key:
        _write_evaluations(view, gt_field, report, options, eval_key)
    return report
