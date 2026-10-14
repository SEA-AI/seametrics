import numpy as np
import pytest

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
        {
            "boxes": np.array([]),
            "scores": np.array([]),
            "labels": np.array([]),
        }
    ]
    target = [
        {
            "boxes": np.array([]),
            "labels": np.array([]),
        }
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
        {
            "boxes": np.array([[258.0, 41.0, 606.0, 285.0]]),
            "scores": np.array([0.7]),
            "labels": np.array([0]),
        }
    ]
    target = [
        {
            "boxes": np.array([]),
            "labels": np.array([]),
        }
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
        {
            "boxes": np.array([]),
            "scores": np.array([]),
            "labels": np.array([]),
        }
    ]
    target = [
        {
            "boxes": np.array([[258.0, 41.0, 606.0, 285.0]]),
            "labels": np.array([0]),
        }
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
        class_agnostic=False,
    )

    predictions = [
        {
            "boxes": np.array(
                [
                    [449.3, 197.75390625, 6.25, 7.03125],
                    [334.3, 181.58203125, 11.5625, 6.85546875],
                ]
            ),
            "labels": np.array([2, 0]),
            "scores": np.array([0.153076171875, 0.72314453125]),
        }
    ]

    references = [
        {
            "boxes": np.array(
                [
                    [449.3, 197.75390625, 6.25, 7.03125],
                    [334.3, 181.58203125, 11.5625, 6.85546875],
                ]
            ),
            "labels": np.array([1, 0]),  # first detection has wrong label
            "area": np.array([132.2, 83.8]),
        }
    ]

    metric.update(predictions, references)
    results = metric.compute()

    val = results["metrics"]["small"]
    assert val["iouThr"] == "1e-10"  # records the threshold exactly
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

    val = results["metrics"]["medium"]
    assert val["iouThr"] == "1e-10"  # records the threshold exactly
    assert val["maxDets"] == 100
    assert np.all(val["tp"] == np.array([1, 0, 0]))  # 1 true positive in class 0
    assert np.all(val["fp"] == np.array([0, 0, 1]))  # 1 false positive in class 2
    assert np.all(val["fn"] == np.array([0, 1, 0]))  # 1 false negative in class 1
    assert np.all(val["duplicates"] == np.array([0, 0, 0]))
    # precision is 1 for class 0, 0 for class 2, undefined for class 1 (divide by 0)
    assert np.all(val["precision"] == np.array([1, -1, 0]))
    assert np.all(val["recall"] == np.array([1, 0, -1]))  # same reasoning
    assert np.all(val["f1"] == np.array([1, -1, -1]))  # only defined if prec+rec > 0
    assert np.all(val["support"] == np.array([1, 1, 0]))
    assert np.all(val["fpi"] == np.array([0, 0, 1]))
    assert val["nImgs"] == 1

    val = results["metrics"]["all"]
    # same as for medium
    assert val["iouThr"] == "1e-10"  # records the threshold exactly
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
        class_agnostic=False,
    )

    predictions = [
        {
            "boxes": np.array(
                [
                    [449.3, 197.75390625, 100.25, 50.03125],
                    [334.3, 181.58203125, 11.5625, 6.85546875],
                ]
            ),
            "labels": np.array([2, 0]),
            "scores": np.array([0.153076171875, 0.72314453125]),
        }
    ]

    references = [
        {
            "boxes": np.array(
                [
                    [449.3, 197.75390625, 6.25, 7.03125],
                    [334.3, 181.58203125, 11.5625, 6.85546875],
                ]
            ),
            "labels": np.array(
                [1, 0]
            ),  # first detection has wrong label & different area category
            "area": np.array([132.2, 83.8]),
        }
    ]

    metric.update(predictions, references)
    results = metric.compute()

    val = results["metrics"]["medium"]
    assert val["iouThr"] == "1e-10"  # records the threshold exactly
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

    val = results["metrics"]["large"]
    # same as for medium
    assert val["iouThr"] == "1e-10"  # records the threshold exactly
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
        labels=[
            0,
            1,
            2,
            3,
        ],  # predefined label list is provided, gt & preds contain only labels 0,2,3
    )
    predictions = [
        {
            "boxes": np.array(
                [
                    [449.3, 197.75390625, 100.25, 50.03125],
                    [334.3, 181.58203125, 11.5625, 6.85546875],
                ]
            ),
            "labels": np.array([2, 0]),
            "scores": np.array([0.153076171875, 0.72314453125]),
        }
    ]

    references = [
        {
            "boxes": np.array(
                [
                    [449.3, 197.75390625, 6.25, 7.03125],
                    [334.3, 181.58203125, 11.5625, 6.85546875],
                ]
            ),
            "labels": np.array([3, 0]),
            "area": np.array([132.2, 83.8]),
        }
    ]
    metric.update(predictions, references)
    results = metric.compute()
    for key, val in results["metrics"].items():
        assert key == "all", "test 1"
        assert val["iouThr"] == "1e-10", "test 1"  # records the threshold exactly
        assert val["maxDets"] == 100, "test 2"
        assert np.all(val["tp"] == np.array([1, 0, 0, 0])), (
            f"{val['tp']}"
        )  # one entry per predefined label, including the absent label 1
        assert np.all(val["fp"] == np.array([0, 0, 1, 0])), "test 4"
        assert np.all(val["fn"] == np.array([0, 0, 0, 1])), "test 5"
        assert np.all(val["duplicates"] == np.array([0, 0, 0, 0])), "test 6"
        assert np.all(val["precision"] == np.array([1, -1, 0, -1])), "test 7"
        assert np.all(val["recall"] == np.array([1, -1, -1, 0])), "test 8"
        assert np.all(val["f1"] == np.array([1, -1, -1, -1])), "test 9"
        assert np.all(val["support"] == np.array([1, 0, 0, 1])), "test 10"
        assert np.all(val["fpi"] == np.array([0, 0, 1, 0])), "test 11"
        assert val["nImgs"] == 1, "test 12"


def test_low_iou_threshold_is_recorded_exactly():
    """Regression: '{:0.2f}' rounded any threshold below 0.005 to "0.00".

    A report run at 1e-5 recorded the same string as one run at 0, and
    results_to_df parsed it back as 0.0, so the reported threshold no longer
    matched the one that produced the numbers.
    """
    preds = [
        {
            "boxes": np.array([[10.0, 10.0, 20.0, 20.0]]),
            "scores": np.array([0.9]),
            "labels": np.array([0]),
        }
    ]
    target = [{"boxes": np.array([[10.0, 10.0, 20.0, 20.0]]), "labels": np.array([0])}]

    for threshold, expected in [
        (0.5, "0.50"),  # familiar two-decimal form is unchanged
        (0.05, "0.05"),
        (1e-5, "1e-05"),
        (1e-10, "1e-10"),
    ]:
        metric = PrecisionRecallF1Support(box_format="xywh", iou_thresholds=[threshold])
        metric.update(preds, target)
        recorded = metric.compute()["metrics"]["all"]["iouThr"]
        assert recorded == expected, f"{threshold} recorded as {recorded}"
        assert np.isclose(float(recorded), threshold, rtol=1e-12, atol=0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"box_format": "xyz"}, "`box_format` to be one of"),
        ({"iou_type": "mask"}, "`iou_type` to be one of"),
        ({"iou_thresholds": 0.5}, "`iou_thresholds` to be `None` or a list of floats"),
        ({"rec_thresholds": 0.5}, "`rec_thresholds` to be `None` or a list of floats"),
        ({"max_detection_thresholds": 100}, "`max_detection_thresholds` .* ints"),
        ({"area_ranges": (0, 1)}, "`area_ranges` to be `None` or a list of lists"),
        ({"area_ranges": [[0, 1, 2]]}, "lists of length 2"),
        ({"area_ranges_labels": ["all"]}, "if `area_ranges` is not provided"),
        (
            {"area_ranges": [[0, 1]], "area_ranges_labels": "all"},
            "`area_ranges_labels` to be `None` or a list of strings",
        ),
        (
            {"area_ranges": [[0, 1]], "area_ranges_labels": ["a", "b"]},
            "a list of length 1",
        ),
        ({"class_agnostic": 1}, "`class_agnostic` to be a boolean"),
        ({"class_agnostic": False, "labels": 0}, "`labels` to be a list"),
        ({"labels": [0]}, "labels to be None if argument `class_agnostic` is True"),
        ({"debug": 1}, "`debug` to be a boolean"),
    ],
)
def test_invalid_constructor_argument_raises(kwargs, match):
    with pytest.raises(ValueError, match=match):
        PrecisionRecallF1Support(**kwargs)


@pytest.mark.parametrize(
    ("pred", "match"),
    [
        # five values per box: box_format="xywh" keeps the columns untouched
        (
            {
                "boxes": np.array([[1.0, 1.0, 2.0, 2.0, 9.0]]),
                "scores": np.array([0.5]),
                "labels": np.array([0]),
            },
            "expected 4 values, got 5",
        ),
        (
            {
                "boxes": np.array([[1.0, 1.0, 2.0, 2.0]]),
                "scores": np.array([0.5]),
                "labels": np.array([0.0]),
            },
            "expected value of type integer",
        ),
        (
            {
                "boxes": np.array([[1.0, 1.0, 2.0, 2.0]]),
                "scores": np.array([1]),
                "labels": np.array([0]),
            },
            "expected value of type float",
        ),
    ],
)
def test_malformed_prediction_raises_on_compute(pred, match):
    metric = PrecisionRecallF1Support(box_format="xywh")
    target = {"boxes": np.array([[1.0, 1.0, 2.0, 2.0]]), "labels": np.array([0])}
    metric.update([pred], [target])
    with pytest.raises(ValueError, match=match):
        metric.compute()


def test_five_value_box_is_rejected_on_update():
    # xyxy -> xywh conversion runs in update, before compute's own box check
    metric = PrecisionRecallF1Support()
    pred = {
        "boxes": np.array([[1.0, 1.0, 2.0, 2.0, 9.0]]),
        "scores": np.array([0.5]),
        "labels": np.array([0]),
    }
    target = {"boxes": np.array([[1.0, 1.0, 2.0, 2.0]]), "labels": np.array([0])}
    with pytest.raises(ValueError, match="4 values each"):
        metric.update([pred], [target])
