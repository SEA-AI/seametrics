"""Convert fitted tracking metric instances to per-sequence DataFrames."""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from .constants import OVERALL_LABEL


class _MetricsForDf(Protocol):
    """Structural type for :func:`results_to_df` metric instances."""

    accumulators: dict[str, object]
    comparison_excluded: frozenset[str]
    RESULT_LAYOUT: str

    def compute(self, sequence: list[str] | str | None = ...) -> dict: ...

    def compute_many(self, sequence_list: list[str]) -> dict: ...


def _sequence_list_for_df(
    metrics: _MetricsForDf,
    sequence_list: list[str] | None,
) -> list[str]:
    """Resolve per-sequence rows to include before pooling OVERALL."""
    if sequence_list is not None:
        return sequence_list
    names = list(metrics.accumulators.keys())
    excluded = metrics.comparison_excluded
    if not excluded:
        return names
    return [name for name in names if name not in excluded]


def _scale_metric_row(flat_result: dict, *, layout: str) -> dict:
    """Apply metric-specific scaling to a flat ``{metric: scalar}`` result.

    TrackingMetrics: ``mota`` is scaled x100 and ``motp`` is converted to
    ``(1 - motp) x 100``; all other metrics are left unchanged.
    HOTAMetrics: all metric values (hota, deta, assa, loca) are scaled x100,
    except the ``num_unique_objects`` count.

    Args:
        flat_result: Mapping from metric name to a single scalar value.
        layout: ``"flat"`` for HOTA-style results, ``"nested"`` for MOT.

    Returns:
        New dict with scaling applied.
    """
    if layout == "flat":
        return {
            k: (v if k == "num_unique_objects" else v * 100)
            for k, v in flat_result.items()
        }
    row = dict(flat_result)
    row["mota"] *= 100
    row["motp"] = (1 - row["motp"]) * 100
    return row


def _flatten_result(result: dict, key: str) -> dict:
    """Flatten a ``compute()`` result to a flat ``{metric: scalar}`` mapping.

    Nested motmetrics-style results (TrackingMetrics and HOTA ``compute_many``)
    map *key* to each metric's inner entry — ``OVERALL_LABEL`` for the pooled
    row, or a sequence name for a per-sequence row.

    Args:
        result: Raw dict returned by ``metrics.compute(...)`` or
            ``metrics.compute_many(...)``.
        key: Inner key to select for nested results (sequence name or
            ``OVERALL_LABEL``).

    Returns:
        Flat ``{metric: scalar}`` dict.
    """
    return {k: v[key] for k, v in result.items()}


def _compute_table_bundle(metrics: _MetricsForDf, sequence_list: list) -> dict:
    """Return one nested result dict covering every table row."""
    if metrics.RESULT_LAYOUT == "flat":
        return metrics.compute_many(sequence_list)
    return metrics.compute(sequence=sequence_list)


def results_to_df(
    metrics: _MetricsForDf, sequence_list: list | None = None
) -> pd.DataFrame:
    """Convert TrackingMetrics or HOTAMetrics results to a DataFrame.

    One batched ``compute()`` / ``compute_many()`` call returns all per-sequence
    rows and the pooled OVERALL row. Appends a pooled OVERALL row — the
    MOT-standard dataset-level score, not a mean of per-sequence values.

    Args:
        metrics: Fitted metric instance with ``accumulators`` and ``compute()``.
        sequence_list: Sequence names to include. Defaults to all accumulators
            minus :attr:`~TrackingMetrics.comparison_excluded` (set by
            :func:`~seametrics.tracking.utils.compute_all_metrics_by_sequence`
            for fair cross-model pooling).

    Raises:
        ValueError: If *sequence_list* contains the reserved ``OVERALL`` label.
    """
    sequence_list = _sequence_list_for_df(metrics, sequence_list)

    if not sequence_list:
        return pd.DataFrame()

    if OVERALL_LABEL in sequence_list:
        raise ValueError(f"{OVERALL_LABEL!r} is reserved for pooled results")

    layout = metrics.RESULT_LAYOUT
    bundle = _compute_table_bundle(metrics, sequence_list)
    rows = []
    for sequence in sequence_list:
        row = _scale_metric_row(_flatten_result(bundle, key=sequence), layout=layout)
        row["sequence"] = sequence
        rows.append(row)
    row = _scale_metric_row(_flatten_result(bundle, key=OVERALL_LABEL), layout=layout)
    row["sequence"] = OVERALL_LABEL
    rows.append(row)

    return pd.DataFrame(rows)


def hota_results_to_df(
    metrics: _MetricsForDf, sequence_list: list | None = None
) -> pd.DataFrame:
    """Alias for results_to_df for backward compatibility.

    Args:
        metrics: Fitted metric instance (see :func:`results_to_df`).
        sequence_list: Optional list of sequence names to include.

    Returns:
        DataFrame with one row per sequence and one column per metric value.
    """
    return results_to_df(metrics, sequence_list)
