from seametrics.detection.imports import _TORCHMETRICS_AVAILABLE

if _TORCHMETRICS_AVAILABLE:
    from seametrics.detection.tm.pr_rec_f1 import PrecisionRecallF1Support
else:
    from seametrics.detection.np.pr_rec_f1 import PrecisionRecallF1Support

from seametrics.detection.det_metrics import DetectionMetrics
from seametrics.detection.utils import (
    compute_all_metrics_by_sequence,
    det_metrics_to_df,
)

__all__ = [
    "_TORCHMETRICS_AVAILABLE",
    "DetectionMetrics",
    "PrecisionRecallF1Support",
    "compute_all_metrics_by_sequence",
    "det_metrics_to_df",
]
