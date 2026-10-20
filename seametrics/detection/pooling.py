"""Pool per-sequence detection results into one overall result."""

from typing import Dict, List

import numpy as np

# Key under which the pooled result is stored alongside the per-sequence results.
OVERALL_KEY = "OVERALL"

# Counts that add across sequences; ratios are recomputed from their sums.
_ADDITIVE_KEYS = ("tp", "fp", "fn", "duplicates", "fpi", "nImgs")


def _pool_area_range_metrics(entries: List[dict]) -> dict:
    """Pool one area range's per-sequence metric dicts into a single dict.

    Counts are summed; precision, recall and f1 are recomputed from the sums with
    the ``-1`` sentinel rules of ``COCOeval._summarize_pr_rec_f1``.

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

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp == 0, -1.0, tp / (tp + fp))
        recall = np.where(tp + fn == 0, -1.0, tp / (tp + fn))
        f1 = np.where(
            (precision == -1) | (recall == -1) | (precision + recall == 0),
            -1.0,
            2 * precision * recall / (precision + recall),
        )

    pooled = {**totals, "support": tp + fn}
    pooled.update(precision=precision, recall=recall, f1=f1)
    result = dict(entries[0])  # keeps COCOeval's key order, range and iouThr
    for key, value in pooled.items():
        result[key] = value.item() if value.ndim == 0 else value
    return result


def aggregate_sequence_results(sequence_results: Dict[str, dict]) -> dict:
    """Pool per-sequence detection results into one overall result.

    Matching never crosses a frame boundary, so counts are additive across
    sequences and the result equals one evaluation of every frame. Each area range
    is pooled on its own: "small" + "large" is not "all", because ground truth
    outside a range is ignored, not counted.

    Args:
        sequence_results (Dict[str, dict]): ``{sequence_name: results}`` as
            returned by :func:`payload_to_det_metrics_by_sequence`. Any existing
            ``"OVERALL"`` entry is skipped so the function is idempotent.

    Returns:
        dict: ``{"metrics": ...}`` shaped like ``PrecisionRecallF1Support.compute()``;
            ``params`` and ``eval`` do not pool.

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
