from .imports import _TORCHMETRICS_AVAILABLE

if _TORCHMETRICS_AVAILABLE:
    from .tm.pr_rec_f1 import PrecisionRecallF1Support
else:
    from .np.pr_rec_f1 import PrecisionRecallF1Support