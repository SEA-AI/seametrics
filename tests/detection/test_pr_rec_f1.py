import numpy as np
from seametrics.detection import PrecisionRecallF1Support

def test_no_images_with_default_args():
    metric = PrecisionRecallF1Support()
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


def test_no_preds_and_target_with_default_args():
    metric = PrecisionRecallF1Support()
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


def test_empty_gt_false_pred_with_default_args():
    metric = PrecisionRecallF1Support()
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


def test_empty_pred_missed_gt_with_default_args():
    metric = PrecisionRecallF1Support()
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

def test_pred_with_wrong_classification_same_area_class_specific():
    area_ranges_tuples = [
        ["all", [0, 1e5**2]],
        ["small", [0**2, 6**2]],
        ["medium", [6**2, 12**2]],
        ["large", [12**2, 1e5**2]],
    ]
    metric = PrecisionRecallF1Support(
        iou_thresholds=[1e-10],
        box_format="xywh",
        area_ranges=[v for _, v in area_ranges_tuples],
        area_ranges_labels=[k for k, v in area_ranges_tuples],
        class_agnostic=False
    )

    predictions = [
        {
            "boxes": np.array([[449.3, 197.75390625, 6.25, 7.03125], [334.3, 181.58203125, 11.5625, 6.85546875]]),
            "labels": np.array([2, 0]),
            "scores": np.array([0.153076171875, 0.72314453125]),
        }
    ]

    references = [
        {
            "boxes": np.array([[449.3, 197.75390625, 6.25, 7.03125], [334.3, 181.58203125, 11.5625, 6.85546875]]),
            "labels": np.array([1, 0]), # first detection has wrong label
            "area": np.array([132.2, 83.8]),
        }
    ]

    metric.update(predictions, references)
    results = metric.compute()
    small_res = results["metrics"].copy()
    small_res.pop("medium")
    small_res.pop("large")
    small_res.pop("all")
    medium_res = results["metrics"].copy()
    medium_res.pop("small")
    medium_res.pop("large")
    medium_res.pop("all")
    all_res = results["metrics"].copy()
    all_res.pop("small")
    all_res.pop("medium")
    all_res.pop("large")

    for key, val in small_res.items():
        assert val["iouThr"] == "0.00"
        assert val["maxDets"] == 100
        assert np.all(val["tp"] == np.array([0, 0, 0]))
        assert np.all(val["fp"] == np.array([0, 0, 0]))
        assert np.all(val["fn"] == np.array([0, 0, 0]))
        assert np.all(val["duplicates"] == np.array([0, 0, 0]))
        assert np.all(val["precision"] == np.array([-1, -1, -1]))
        assert np.all(val["recall"] == np.array([-1, -1, -1]))
        assert np.all(val["f1"] == np.array([-1, -1, -1]))
        assert np.all(val["support"] == np.array([0, 0, 0]))
        assert np.all(val["fpi"] == np.array([0, 0, 0]))
        assert val["nImgs"] == 1

    for key, val in medium_res.items():
        assert val["iouThr"] == "0.00"
        assert val["maxDets"] == 100
        assert np.all(val["tp"] == np.array([1, 0, 0])) # 1 true positive in class 0
        assert np.all(val["fp"] == np.array([0, 0, 1])) # 1 false positive in class 2
        assert np.all(val["fn"] == np.array([0, 1, 0])) # 1 false negative in class 1
        assert np.all(val["duplicates"] == np.array([0, 0, 0]))
        assert np.all(val["precision"] == np.array([1, -1, 0])) # precision is 1 for class 0, 0 for class 2, and undefined for class 1 (divide by 0)
        assert np.all(val["recall"] == np.array([1, 0, -1])) # same reasoning as for precision
        assert np.all(val["f1"] == np.array([1, -1, -1])) # onlly defined if (prec+rec) > 0
        assert np.all(val["support"] == np.array([1, 1, 0])) 
        assert np.all(val["fpi"] == np.array([0, 0, 1]))
        assert val["nImgs"] == 1

    for key, val in all_res.items():
        # same as for medium
        assert val["iouThr"] == "0.00"
        assert val["maxDets"] == 100
        assert np.all(val["tp"] == np.array([1, 0, 0]))
        assert np.all(val["fp"] == np.array([0, 0, 1]))
        assert np.all(val["fn"] == np.array([0, 1, 0]))
        assert np.all(val["duplicates"] == np.array([0, 0, 0]))
        assert np.all(val["precision"] == np.array([1, -1, 0]))
        assert np.all(val["recall"] == np.array([1, 0, -1]))
        assert np.all(val["f1"] == np.array([1, -1, -1]))
        assert np.all(val["support"] == np.array([1, 1, 0]))
        assert np.all(val["fpi"] == np.array([0, 0, 1]))
        assert val["nImgs"] == 1

def test_pred_with_wrong_classification_different_area_class_specific():
    area_ranges_tuples = [
        ["all", [0, 1e5**2]],
        ["small", [0**2, 6**2]],
        ["medium", [6**2, 12**2]],
        ["large", [12**2, 1e5**2]],
    ]
    metric = PrecisionRecallF1Support(
        iou_thresholds=[1e-10],
        box_format="xywh",
        area_ranges=[v for _, v in area_ranges_tuples],
        area_ranges_labels=[k for k, v in area_ranges_tuples],
        class_agnostic=False
    )

    predictions = [
        {
            "boxes": np.array([[449.3, 197.75390625, 100.25, 50.03125], [334.3, 181.58203125, 11.5625, 6.85546875]]),
            "labels": np.array([2, 0]),
            "scores": np.array([0.153076171875, 0.72314453125]),
        }
    ]

    references = [
        {
            "boxes": np.array([[449.3, 197.75390625, 6.25, 7.03125], [334.3, 181.58203125, 11.5625, 6.85546875]]),
            "labels": np.array([1, 0]), # first detection has wrong label & different area category
            "area": np.array([132.2, 83.8]),
        }
    ]

    metric.update(predictions, references)
    results = metric.compute()
    medium_res = results["metrics"].copy()
    medium_res.pop("small")
    medium_res.pop("large")
    medium_res.pop("all")
    large_res = results["metrics"].copy()
    large_res.pop("small")
    large_res.pop("medium")
    large_res.pop("all")


    for key, val in medium_res.items():
        assert val["iouThr"] == "0.00"
        assert val["maxDets"] == 100
        assert np.all(val["tp"] == np.array([1, 0, 0]))
        assert np.all(val["fp"] == np.array([0, 0, 0]))
        assert np.all(val["fn"] == np.array([0, 1, 0]))
        assert np.all(val["duplicates"] == np.array([0, 0, 0]))
        assert np.all(val["precision"] == np.array([1, -1, -1]))
        assert np.all(val["recall"] == np.array([1, 0, -1]))
        assert np.all(val["f1"] == np.array([1, -1, -1]))
        assert np.all(val["support"] == np.array([1, 1, 0]))
        assert np.all(val["fpi"] == np.array([0, 0, 0]))
        assert val["nImgs"] == 1

    for key, val in large_res.items():
        # same as for medium
        assert val["iouThr"] == "0.00"
        assert val["maxDets"] == 100
        assert np.all(val["tp"] == np.array([0, 0, 0]))
        assert np.all(val["fp"] == np.array([0, 0, 1]))
        assert np.all(val["fn"] == np.array([0, 0, 0]))
        assert np.all(val["duplicates"] == np.array([0, 0, 0]))
        assert np.all(val["precision"] == np.array([-1, -1, 0]))
        assert np.all(val["recall"] == np.array([-1, -1, -1]))
        assert np.all(val["f1"] == np.array([-1, -1, -1]))
        assert np.all(val["support"] == np.array([0, 0, 0]))
        assert np.all(val["fpi"] == np.array([0, 0, 1]))
        assert val["nImgs"] == 1

def test_pred_with_predefined_labels_class_specific():
    metric = PrecisionRecallF1Support(
        iou_thresholds=[1e-10],
        class_agnostic=False,
        box_format="xywh",
        labels=[0, 1, 2, 3] # predefined label list is provided, gt & preds contain only labels 0,2,3
    )
    predictions = [
        {
            "boxes": np.array([[449.3, 197.75390625, 100.25, 50.03125], [334.3, 181.58203125, 11.5625, 6.85546875]]),
            "labels": np.array([2, 0]),
            "scores": np.array([0.153076171875, 0.72314453125]),
        }
    ]

    references = [
        {
            "boxes": np.array([[449.3, 197.75390625, 6.25, 7.03125], [334.3, 181.58203125, 11.5625, 6.85546875]]),
            "labels": np.array([3, 0]),
            "area": np.array([132.2, 83.8]),
        }
    ]
    metric.update(predictions, references)
    results = metric.compute()
    for key, val in results["metrics"].items():
        assert key == "all", "test 1"
        assert val["iouThr"] == "0.00", "test 1"
        assert val["maxDets"] == 100, "test 2"
        assert np.all(val["tp"] == np.array([1, 0, 0, 0])), f"{val['tp']}" # results contain all 4 provided classes, and add pred & gt accordingly
        assert np.all(val["fp"] == np.array([0, 0, 1, 0])), "test 4"
        assert np.all(val["fn"] == np.array([0, 0, 0, 1])), "test 5"
        assert np.all(val["duplicates"] == np.array([0, 0, 0, 0])), "test 6"
        assert np.all(val["precision"] == np.array([1, -1, 0, -1])), "test 7"
        assert np.all(val["recall"] == np.array([1, -1, -1, 0])), "test 8"
        assert np.all(val["f1"] == np.array([1, -1, -1, -1])), "test 9"
        assert np.all(val["support"] == np.array([1, 0, 0, 1])), "test 10"
        assert np.all(val["fpi"] == np.array([0, 0, 1, 0])), "test 11"
        assert val["nImgs"] == 1, "test 12"
