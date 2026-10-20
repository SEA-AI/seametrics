import fiftyone as fo
import numpy as np
import pytest

from seametrics.detection.utils import (
    _validate_arrays,
    frame_dets_to_det_metrics,
    get_relevant_fields,
    get_values,
    payload_sequence_to_det_metrics,
    prepare_data_for_det_metrics,
)


def assert_det_dicts(dict1, dict2):
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())
    assert keys1 == keys2, f"Keys mismatch: {keys1 ^ keys2}"

    for key in dict1:
        assert np.array_equal(dict1[key], dict2[key]), f"Value mismatch for key {key}"


def assert_list_of_det_dicts(list1, list2):
    assert len(list1) == len(list2), "Length mismatch"
    for dict1, dict2 in zip(list1, list2, strict=True):
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
    desired_outputs = [
        {
            "boxes": np.array([]),
            "scores": np.array([]),
            "labels": np.array([]),
        }
    ]
    outputs = payload_sequence_to_det_metrics(sequence_dets, w, h)
    assert_list_of_det_dicts(outputs, desired_outputs)


def test_payload_sequence_to_det_metrics_with_empty_ground_truth():
    sequence_dets = [[]]
    w = 640
    h = 512
    desired_outputs = [
        {
            "boxes": np.array([]),
            "labels": np.array([]),
            "area": np.array([]),
        }
    ]
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
        }
        for _ in range(10)
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
        }
        for _ in range(10)
    ]
    outputs = payload_sequence_to_det_metrics(sequence_dets, w, h, is_gt=True)
    assert_list_of_det_dicts(outputs, desired_outputs)


def test_prepare_data_for_det_metrics_denormalizes_and_converts():
    # normalized xywh (0.1, 0.2, 0.3, 0.4) on 100x50 -> xywh (10, 10, 30, 20)
    # -> xyxy (10, 10, 40, 30)
    gt = [[[0.1, 0.2, 0.3, 0.4]], [[0.5, 0.5, 0.1, 0.2]]]
    dt = [[[0.1, 0.2, 0.3, 0.4]], []]
    target, preds = prepare_data_for_det_metrics(
        gt, [["boat"], ["buoy"]], dt, [["boat"], []], [[None], []], 100, 50
    )
    assert_list_of_det_dicts(
        target,
        [
            {"boxes": np.array([[10.0, 10.0, 40.0, 30.0]]), "labels": np.array([0])},
            {"boxes": np.array([[50.0, 25.0, 60.0, 35.0]]), "labels": np.array([0])},
        ],
    )
    # all scores None -> every prediction scored 1.0
    assert_list_of_det_dicts(
        preds,
        [
            {
                "boxes": np.array([[10.0, 10.0, 40.0, 30.0]]),
                "labels": np.array([0]),
                "scores": np.array([1.0]),
            },
            {"boxes": np.array([]), "labels": np.array([]), "scores": np.array([])},
        ],
    )


class _View:
    media_type = "group"


@pytest.mark.parametrize("fn", [get_relevant_fields, get_values])
def test_fiftyone_helpers_reject_unknown_media_type(fn):
    with pytest.raises(ValueError, match="Unsupported media type: group"):
        fn(_View(), "gt")


def test_validate_arrays_wraps_a_single_frame():
    # one frame of two boxes becomes a one-frame list
    frames = _validate_arrays([[0, 0, 1, 1], [0, 0, 2, 2]], data_type="bbox")
    assert len(frames) == 1
    np.testing.assert_array_equal(frames[0], [[0, 0, 1, 1], [0, 0, 2, 2]])
    assert _validate_arrays([0.5, 0.7], data_type="score")[0].tolist() == [0.5, 0.7]
    assert _validate_arrays(None, data_type="label")[0].size == 0


def test_validate_arrays_rejects_unknown_data_type():
    with pytest.raises(ValueError, match="Unsupported data type: keypoint"):
        _validate_arrays([[0, 0, 1, 1]], data_type="keypoint")
