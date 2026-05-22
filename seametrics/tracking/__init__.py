"""Tracking metrics: MOT (TrackingMetrics) and HOTA (HOTAMetrics)."""

from .hota import HOTAMetrics
from .imports import _MOTMETRICS_AVAILABLE
from .report import build_comparison_html
from .track import TrackingMetrics

__all__ = [
    "_MOTMETRICS_AVAILABLE",
    "HOTAMetrics",
    "TrackingMetrics",
    "build_comparison_html",
]
