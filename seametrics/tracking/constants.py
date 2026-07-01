"""Shared constants for tracking metrics."""

import numpy as np

#: Label used for the pooled, dataset-level aggregate row appended by
#: :func:`~seametrics.tracking.results_df.results_to_df`. Matches the name
#: ``motmetrics`` uses for its overall row so MOT and HOTA DataFrames stay
#: consistent.
OVERALL_LABEL = "OVERALL"

#: Metric names aggregated by summation across sequences (integer counts), as
#: opposed to ratio/derived metrics which are pooled. Shared so the metric
#: classes and the report agree on a single source of truth.
COUNT_METRICS = (
    "num_frames",
    "mostly_tracked",
    "partially_tracked",
    "mostly_lost",
    "num_switches",
    "num_false_positives",
    "num_misses",
    "num_fragmentations",
    "num_unique_objects",
)

#: Ratio/derived metrics (pooled across sequences), as opposed to COUNT_METRICS.
RATIO_METRICS = ("mota", "motp", "idf1", "idp", "idr", "precision", "recall")

#: Column width for MOT tracker-format arrays produced by
#: :func:`~seametrics.tracking.utils.prepare_data_for_det_metrics`.
TRACKER_ARRAY_COLS = 10

#: IoU thresholds for HOTA evaluation (19 values: 0.05 … 0.95).
HOTA_THRESHOLDS = np.arange(0.05, 0.95 + 1e-9, 0.05)

#: Human-readable metric class names for comparison reports.
DISPLAY_NAMES = {"TrackingMetrics": "MOT Metrics", "HOTAMetrics": "HOTA Metrics"}

#: Default model colours for comparison report charts.
MODEL_COLORS = [
    "rgba(233,69,96,0.8)",
    "rgba(52,168,235,0.8)",
    "rgba(52,211,153,0.8)",
    "rgba(251,191,36,0.8)",
    "rgba(167,139,250,0.8)",
    "rgba(251,146,60,0.8)",
]
