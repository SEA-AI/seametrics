"""Pool per-sequence detection results into one overall result."""

from typing import Dict, List, Union

import numpy as np

# Key under which the pooled result is stored alongside the per-sequence results.
OVERALL_KEY = "OVERALL"

# Count fields that are additive across sequences. Ratios (precision/recall/f1)
# are NOT additive and must be recomputed from the pooled counts.
_ADDITIVE_KEYS = ("tp", "fp", "fn", "duplicates", "fpi", "nImgs")


def _as_scalar_or_array(value: np.ndarray, cast: type) -> Union[int, float, np.ndarray]:
    """Return a python scalar for 0-d input, else the array, like COCOeval."""
    return cast(value) if value.ndim == 0 else value


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
