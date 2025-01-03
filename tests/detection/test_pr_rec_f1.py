from typing import Type
import pytest
import numpy as np
from seametrics.detection.np.pr_rec_f1 import PrecisionRecallF1Support as PRF1Np

backends = [PRF1Np]

# Try importing the torchmetrics-based backend
try:
    from seametrics.detection.tm.pr_rec_f1 import PrecisionRecallF1Support as PRF1Tm

    backends.append(PRF1Tm)
except ImportError:
    print("TorchMetrics not installed, skipping corresponding tests.")


@pytest.mark.parametrize("MetricClass", backends)
def test_no_images_with_default_args(MetricClass: Type[PRF1Np]):
    metric = MetricClass()
    preds = []
    target = []
    metric.update(preds, target)
    results = metric.compute()
    for key, val in results["metrics"].items():
        assert key == "all"
        assert val["iouThr"] == "0.50"
        assert val["maxDets"] == 100
        assert val["tp"] == 0
        assert val["fp"] == 0
        assert val["fn"] == 0
        assert val["duplicates"] == 0
        assert val["precision"] == -1
        assert val["recall"] == -1
        assert val["f1"] == -1
        assert val["support"] == 0
        assert val["fpi"] == 0
        assert val["nImgs"] == 0


@pytest.mark.parametrize("MetricClass", backends)
def test_no_preds_and_target_with_default_args(MetricClass: Type[PRF1Np]):
    metric = MetricClass()
    preds = [
        dict(
            boxes=np.array([]),
            scores=np.array([]),
            labels=np.array([]),
        )
    ]
    target = [
        dict(
            boxes=np.array([]),
            labels=np.array([]),
        )
    ]
    metric.update(preds, target)
    results = metric.compute()
    for key, val in results["metrics"].items():
        assert key == "all"
        assert val["iouThr"] == "0.50"
        assert val["maxDets"] == 100
        assert val["tp"] == 0
        assert val["fp"] == 0
        assert val["fn"] == 0
        assert val["duplicates"] == 0
        assert val["precision"] == -1
        assert val["recall"] == -1
        assert val["f1"] == -1
        assert val["support"] == 0
        assert val["fpi"] == 0
        assert val["nImgs"] == 1


@pytest.mark.parametrize("MetricClass", backends)
def test_empty_gt_false_pred_with_default_args(MetricClass: Type[PRF1Np]):
    metric = MetricClass()
    preds = [
        dict(
            boxes=np.array([[258.0, 41.0, 606.0, 285.0]]),
            scores=np.array([0.7]),
            labels=np.array([0]),
        )
    ]
    target = [
        dict(
            boxes=np.array([]),
            labels=np.array([]),
        )
    ]
    metric.update(preds, target)
    results = metric.compute()
    for key, val in results["metrics"].items():
        assert key == "all"
        assert val["iouThr"] == "0.50"
        assert val["maxDets"] == 100
        assert val["tp"] == 0
        assert val["fp"] == 1
        assert val["fn"] == 0
        assert val["duplicates"] == 0
        assert val["precision"] == 0
        assert val["recall"] == -1
        assert val["f1"] == -1
        assert val["support"] == 0
        assert val["fpi"] == 1
        assert val["nImgs"] == 1


@pytest.mark.parametrize("MetricClass", backends)
def test_empty_pred_missed_gt_with_default_args(MetricClass: Type[PRF1Np]):
    metric = MetricClass()
    preds = [
        dict(
            boxes=np.array([]),
            scores=np.array([]),
            labels=np.array([]),
        )
    ]
    target = [
        dict(
            boxes=np.array([[258.0, 41.0, 606.0, 285.0]]),
            labels=np.array([0]),
        )
    ]
    metric.update(preds, target)
    results = metric.compute()
    for key, val in results["metrics"].items():
        assert key == "all"
        assert val["iouThr"] == "0.50"
        assert val["maxDets"] == 100
        assert val["tp"] == 0
        assert val["fp"] == 0
        assert val["fn"] == 1
        assert val["duplicates"] == 0
        assert val["precision"] == -1
        assert val["recall"] == 0
        assert val["f1"] == -1
        assert val["support"] == 1
        assert val["fpi"] == 0
        assert val["nImgs"] == 1
