"""DetectionMetrics: sequence-aware wrapper around PrecisionRecallF1Support.

Mirrors the TrackingMetrics / HOTAMetrics interface so detection results can be
fed into the same compute_all_metrics_by_sequence pipeline and
build_comparison_html report.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from seametrics.detection.imports import _TORCHMETRICS_AVAILABLE

if _TORCHMETRICS_AVAILABLE:
    from seametrics.detection.tm.pr_rec_f1 import PrecisionRecallF1Support
else:
    from seametrics.detection.np.pr_rec_f1 import PrecisionRecallF1Support

if TYPE_CHECKING:
    import numpy as np

_METRIC_KEYS = (
    "precision", "recall", "f1",
    "tp", "fp", "fn", "duplicates", "support", "fpi",
)
_NAN_ROW = {k: float("nan") for k in (*_METRIC_KEYS, "n_imgs")}


def _extract_metrics(raw: dict) -> Dict[str, Dict[str, Any]]:
    """Extract the per-area-range metrics from a PrecisionRecallF1Support result.

    Parameters
    ----------
    raw : dict
        Full output of ``PrecisionRecallF1Support.compute()``.

    Returns:
    -------
    dict
        ``{area_range_label: {metric_key: value, ...}}`` with ``n_imgs``
        normalised from the raw ``nImgs`` key.
    """
    out = {}
    for label, m in raw["metrics"].items():
        out[label] = {k: m[k] for k in _METRIC_KEYS}
        out[label]["n_imgs"] = m["nImgs"]
    return out


def _nan_areas(instance: PrecisionRecallF1Support) -> Dict[str, Dict[str, Any]]:
    """Return NaN-filled dicts for every area range label configured on *instance*.

    Parameters
    ----------
    instance : PrecisionRecallF1Support
        Instance whose ``area_ranges_labels`` define the expected keys.

    Returns:
    -------
    dict
        ``{area_range_label: NaN-filled metrics dict}``.
    """
    labels = (
        instance.area_ranges_labels
        if hasattr(instance, "area_ranges_labels")
        else ["all"]
    )
    return {lbl: dict(_NAN_ROW) for lbl in labels}


class DetectionMetrics:
    """Sequence-aware wrapper around PrecisionRecallF1Support.

    Exposes the same ``accumulators`` / ``failed_sequences`` / ``compute`` /
    ``log_failed_sequence`` interface as ``TrackingMetrics`` and
    ``HOTAMetrics``, enabling detection metrics to participate in
    ``compute_all_metrics_by_sequence`` and the shared HTML report.

    Parameters
    ----------
    iou_threshold : float
        IoU threshold passed to the underlying metric (default 0.0001).  The
        underlying ``PrecisionRecallF1Support`` accepts a list of thresholds
        via ``iou_thresholds``; this convenience scalar sets a single
        threshold.
    **kwargs
        Forwarded verbatim to ``PrecisionRecallF1Support``.  Use
        ``area_ranges`` / ``area_ranges_labels`` to configure size-based
        breakdowns (small / medium / large / all).

    Attributes:
    ----------
    accumulators : dict
        ``{sequence_name: PrecisionRecallF1Support}`` — one instance per
        successfully updated sequence.
    failed_sequences : dict
        ``{sequence_name: reason_string}`` — sequences that could not be
        evaluated.
    """

    def __init__(self, iou_threshold: float = 0.0001, **kwargs: object) -> None:
        """Initialise DetectionMetrics.

        Parameters
        ----------
        iou_threshold : float
            Single IoU threshold at which to evaluate (default 0.0001).
        **kwargs
            Passed through to ``PrecisionRecallF1Support``.
        """
        self.accumulators: Dict[str, PrecisionRecallF1Support] = {}
        self.failed_sequences: Dict[str, str] = {}
        self.iou_threshold = iou_threshold
        self._kwargs = {"iou_thresholds": [iou_threshold], **kwargs}
        self._global: PrecisionRecallF1Support = PrecisionRecallF1Support(
            **self._kwargs
        )

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def update(
        self,
        preds: List[Dict[str, np.ndarray]],
        targets: List[Dict[str, np.ndarray]],
        sequence_name: str,
    ) -> None:
        """Accumulate one sequence of frame-level detections.

        Parameters
        ----------
        preds : list of dict
            Per-frame predictions.  Each dict must have ``"boxes"`` (Nx4),
            ``"labels"`` (N,), and ``"scores"`` (N,).
        targets : list of dict
            Per-frame ground-truth.  Each dict must have ``"boxes"`` (Mx4)
            and ``"labels"`` (M,).  ``"area"`` is optional.
        sequence_name : str
            Unique identifier for this sequence; used as the key in
            ``accumulators``.
        """
        try:
            seq_instance = PrecisionRecallF1Support(**self._kwargs)
            seq_instance.update(preds, targets)
            self.accumulators[sequence_name] = seq_instance
            self._global.update(preds, targets)
        except Exception as exc:
            self.log_failed_sequence(sequence_name, targets, preds, exc=exc)

    def compute(self, sequence: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Compute detection metrics for one sequence or across all sequences.

        Parameters
        ----------
        sequence : str, optional
            When provided, compute metrics for the named sequence only.
            When ``None``, compute the global aggregate over all accumulated
            sequences.

        Returns:
        -------
        dict
            ``{area_range_label: {metric_key: value}}`` where metric keys are
            ``precision``, ``recall``, ``f1``, ``tp``, ``fp``, ``fn``,
            ``duplicates``, ``support``, ``fpi``, ``n_imgs``.
        """
        instance = (
            self.accumulators[sequence] if sequence is not None else self._global
        )
        try:
            raw = instance.compute()
            return _extract_metrics(raw)
        except Exception:
            return _nan_areas(instance)

    def log_failed_sequence(
        self,
        sequence_name: str,
        gt: list,
        pred: list,
        exc: Optional[Exception] = None,
    ) -> None:
        """Record why a sequence could not be evaluated.

        Parameters
        ----------
        sequence_name : str
            Name of the sequence that failed.
        gt : list
            Ground-truth data (used only to check emptiness).
        pred : list
            Prediction data (used only to check emptiness).
        exc : Exception, optional
            Exception that caused the failure, if any.
        """
        if len(gt) == 0 and len(pred) == 0:
            reason = "No ground truth and no predictions"
        elif len(gt) == 0:
            reason = "No ground truth"
        elif len(pred) == 0:
            reason = "No predictions"
        elif exc is not None:
            reason = f"{type(exc).__name__}: {exc}"
        else:
            reason = "Missing data from GT or Pred"
        self.failed_sequences[sequence_name] = reason
