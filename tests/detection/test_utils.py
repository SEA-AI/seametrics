import numpy as np
import fiftyone as fo
from seametrics.detection.utils import (
    frame_dets_to_det_metrics,
    payload_sequence_to_det_metrics,
)


def assert_det_dicts(dict1, dict2):
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())
    assert keys1 == keys2, f"Keys mismatch: {keys1 ^ keys2}"

    for key in dict1.keys():
        assert np.array_equal(dict1[key], dict2[key]), f"Value mismatch for key {key}"

def assert_list_of_det_dicts(list1, list2):
    assert len(list1) == len(list2), "Length mismatch"
    for i, (dict1, dict2) in enumerate(zip(list1, list2)):
        assert_det_dicts(dict1, dict2)


def test_frame_dets_to_det_metrics_with_empty_predictions():
    dets = []
    w = 640
    h = 512
    desired_format = {
        "boxes": np.array([]),
        "scores": np.array([]),
        "labels": np.array([]),
    }
    metrics_format = frame_dets_to_det_metrics(dets, w, h)
    assert_det_dicts(metrics_format, desired_format)


def test_frame_dets_to_det_metrics_with_empty_ground_truth():
    dets = []
    w = 640
    h = 512
    desired_format = {
        "boxes": np.array([]),
        "labels": np.array([]),
        "area": np.array([]),
    }
    metrics_format = frame_dets_to_det_metrics(dets, w, h, is_gt=True)
    # check both dicts have the same keys
    assert_det_dicts(metrics_format, desired_format)


def test_frame_dets_to_det_metrics_with_detections():
    dets = [
        fo.Detection(
            label="FAR_AWAY_OBJECT",
            bounding_box=[0, 0, 1, 1],
            confidence=0.153076171875,
        )
    ]
    w = 640
    h = 512
    desired_format = {
        "boxes": np.array([[0, 0, 640, 512]]),
        "scores": np.array([0.153076171875]),
        "labels": np.array([0]),
    }
    metrics_format = frame_dets_to_det_metrics(dets, w, h)
    # check both dicts have the same keys
    assert_det_dicts(metrics_format, desired_format)


def test_frame_dets_to_det_metrics_with_ground_truth():
    dets = [
        fo.Detection(
            label="FAR_AWAY_OBJECT",
            bounding_box=[0, 0, 1, 1],
            area=100,  # fake value, we only care if it is forwarded
        ),
        fo.Detection(
            label="FAR_AWAY_OBJECT",
            bounding_box=[0, 0, 0.5, 0.5],
            area=50,  # fake value, we only care if it is forwarded
        ),
    ]
    w = 640
    h = 512
    desired_format = {
        "boxes": np.array(
            [
                [0, 0, 640, 512],
                [0, 0, 320, 256],
            ]
        ),
        "labels": np.array([0, 0]),
        "area": np.array([100, 50]),
    }
    metrics_format = frame_dets_to_det_metrics(dets, w, h, is_gt=True)
    # check both dicts have the same keys
    assert_det_dicts(metrics_format, desired_format)


def test_payload_sequence_to_det_metrics_with_empty_sequence():
    sequence_dets = []
    w = 640
    h = 512
    desired_output = []
    output = payload_sequence_to_det_metrics(sequence_dets, w, h)
    assert_list_of_det_dicts(output, desired_output)


def test_payload_sequence_to_det_metrics_with_empty_predictions():
    sequence_dets = [[]]
    w = 640
    h = 512
    desired_outputs = [{
        "boxes": np.array([]),
        "scores": np.array([]),
        "labels": np.array([]),
    }]
    outputs = payload_sequence_to_det_metrics(sequence_dets, w, h)
    assert_list_of_det_dicts(outputs, desired_outputs)


def test_payload_sequence_to_det_metrics_with_empty_ground_truth():
    sequence_dets = [[]]
    w = 640
    h = 512
    desired_outputs = [{
        "boxes": np.array([]),
        "labels": np.array([]),
        "area": np.array([]),
    }]
    outputs = payload_sequence_to_det_metrics(sequence_dets, w, h, is_gt=True)
    assert_list_of_det_dicts(outputs, desired_outputs)


def test_payload_sequence_to_det_metrics_with_detections():
    sequence_dets = [
        [
            fo.Detection(
                label="FAR_AWAY_OBJECT",
                bounding_box=[0, 0, 1, 1],
                confidence=0.153076171875,
            )
        ]
    ] * 10
    w = 640
    h = 512
    desired_outputs = [
        {
            "boxes": np.array([[0, 0, 640, 512]]),
            "scores": np.array([0.153076171875]),
            "labels": np.array([0]),
        } for _ in range(10)
    ]
    outputs = payload_sequence_to_det_metrics(sequence_dets, w, h)
    assert_list_of_det_dicts(outputs, desired_outputs)


def test_payload_sequence_to_det_metrics_with_ground_truth():
    sequence_dets = [
        [
            fo.Detection(
                label="FAR_AWAY_OBJECT",
                bounding_box=[0, 0, 1, 1],
                area=100,  # fake value, we only care if it is forwarded
            ),
            fo.Detection(
                label="FAR_AWAY_OBJECT",
                bounding_box=[0, 0, 0.5, 0.5],
                area=50,  # fake value, we only care if it is forwarded
            ),
        ]
    ] * 10
    w = 640
    h = 512
    desired_outputs = [
        {
            "boxes": np.array(
                [
                    [0, 0, 640, 512],
                    [0, 0, 320, 256],
                ]
            ),
            "labels": np.array([0, 0]),
            "area": np.array([100, 50]),
        } for _ in range(10)
    ]
    outputs = payload_sequence_to_det_metrics(sequence_dets, w, h, is_gt=True)
    assert_list_of_det_dicts(outputs, desired_outputs)
