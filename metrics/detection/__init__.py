from metrics.detection.imports import _TORCHMETRICS_AVAILABLE

if _TORCHMETRICS_AVAILABLE:
    from metrics.detection.tm.pr_rec_f1 import PrecisionRecallF1Support
else:
    from metrics.detection.np.pr_rec_f1 import PrecisionRecallF1Support