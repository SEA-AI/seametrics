"""Object detection metrics."""

from seametrics.detection.imports import _TORCHMETRICS_AVAILABLE

if _TORCHMETRICS_AVAILABLE:
    from seametrics.detection.tm.pr_rec_f1 import PrecisionRecallF1Support
else:
    from seametrics.detection.np.pr_rec_f1 import PrecisionRecallF1Support

__all__ = ["PrecisionRecallF1Support"]
