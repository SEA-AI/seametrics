import math
from pprint import pprint

import motmetrics as mm
import numpy as np
import pandas as pd
import pytest

from seametrics.payload import Payload, Resolution, Sequence
from seametrics.user_friendly.utils import (
    get_formated_references,
    get_formated_references,
    payload_to_uf_metrics,
    UFM,
)

ufm = UFM()

def test_recognition():
    """
    Tests the recognition function
    """

    track_ratios = pd.Series(
        [0.2, 0.4, 0.6, 0.8, 1.0],
        index=["OId1", "OId2", "OId3", "OId4", "OId5"],
        name="count",
    )

    th = 0.3
    assert UFM.recognition(track_ratios, th=th) == 4

    th = 0.5
    assert UFM.recognition(track_ratios, th=th) == 3

    th = 0.8
    assert UFM.recognition(track_ratios, th=th) == 2

    th = 1.0
    assert UFM.recognition(track_ratios, th=th) == 1

    th = 1.1
    assert UFM.recognition(track_ratios, th=th) == 0


def test_calculate():
    """
    Tests the calculate function with a typical case of valid metrics.
    """
    predictions = [
        [1, 1, 0.1, 0.2, 0.3, 0.4, 0.9],
        [1, 2, 0.5, 0.6, 0.2, 0.3, 0.8],
    ]
    references = [
        [1, 1, 0.1, 0.2, 0.3, 0.4],
        [1, 2, 0.5, 0.6, 0.2, 0.3],
    ]

    result = ufm.calculate(predictions, references)

    assert "tp" in result
    assert "fn" in result

    assert result["tp"] == 2.0, f"Expected tp to be 2.0, got {result['tp']}"
    assert result["fn"] == 0.0, f"Expected fn to be 0.0, got {result['fn']}"
    assert (
        result["unique_obj_count"] == 2
    ), f"Expected unique_obj_count to be 2, got {result['unique_obj_count']}"
    assert (
        result["mostly_tracked_count_0_3"] == 2
    ), f"Expected mostly_tracked_count_0.3 to be 2, got {result['mostly_tracked_count_0_3']}"
    assert (
        result["mostly_tracked_count_0_5"] == 2
    ), f"Expected mostly_tracked_count_0_5 to be 2, got {result['mostly_tracked_count_0_5']}"
    assert (
        result["mostly_tracked_count_0_8"] == 2
    ), f"Expected mostly_tracked_count_0_8 to be 2, got {result['mostly_tracked_count_0_8']}"


def test_calculate_invalid_shapes():
    # Invalid shape for predictions (missing confidence column)
    """
    Tests the calculate function for cases with invalid shapes for predictions and references.

    Ensures that ValueError is raised when:
    - Predictions are missing the confidence column.
    - References are missing the width and height columns.

    Checks that the appropriate error message is provided for each case.
    """
    predictions = [[1, 1, 0.1, 0.2, 0.3, 0.4]]  # Only 6 columns instead of 7
    references = [[1, 1, 0.1, 0.2, 0.3, 0.4]]

    with pytest.raises(
        ValueError,
        match="The predictions should be a 2D array with 7 columns",
    ):
        ufm.calculate(predictions, references)

    # Invalid shape for references (missing width and height)
    predictions = [[1, 1, 0.1, 0.2, 0.3, 0.4, 0.9]]
    references = [[1, 1, 0.1, 0.2]]  # Only 4 columns instead of 6

    with pytest.raises(
        ValueError,
        match="The references should be a 2D array with 6 columns",
    ):
        ufm.calculate(predictions, references)


def test_calculate_invalid_frame_numbers():
    # Invalid frame number in predictions (frame number is 0)
    """
    Tests the calculate function for cases with invalid frame numbers in predictions and references.

    Ensures that ValueError is raised when:
    - The frame number in the predictions is 0.
    - The frame number in the references is negative.

    Checks that the appropriate error message is provided for each case.
    """
    predictions = [[0, 1, 0.1, 0.2, 0.3, 0.4, 0.9]]
    references = [[1, 1, 0.1, 0.2, 0.3, 0.4]]

    with pytest.raises(
        ValueError,
        match="The frame number in the predictions should be a positive integer",
    ):
        ufm.calculate(predictions, references)

    # Invalid frame number in references (frame number is negative)
    predictions = [[1, 1, 0.1, 0.2, 0.3, 0.4, 0.9]]
    references = [[-1, 1, 0.1, 0.2, 0.3, 0.4]]

    with pytest.raises(
        ValueError,
        match="The frame number in the references should be a positive integer",
    ):
        ufm.calculate(predictions, references)


def test_calculate_single_data_point():
    # Matching case
    """
    Tests the calculate function with a single data point in both predictions and references.

    Verifies the following scenarios:
    - Matching case: prediction and reference have the same frame, object ID, and bounding box,
      expecting a true positive (TP) with no false positives (FP) or false negatives (FN).
    - Non-matching case: prediction and reference have the same frame and object ID but different bounding boxes,
      expecting no true positives (TP), one false positive (FP), and one false negative (FN).

    Asserts that the correct number of unique ground truth IDs is detected in both cases.
    """
    predictions = [[1, 1, 0.1, 0.2, 0.2, 0.2, 0.9]]
    references = [[1, 1, 0.1, 0.2, 0.2, 0.2]]
    result = ufm.calculate(predictions, references)

    assert result["tp"] == 1, "Expected 1 TP for a matching single data point"
    assert result["fn"] == 0, "No FN expected for a matching single data point"
    assert result["unique_obj_count"] == 1, "Expected 1 unique GT ID"

    # Non-matching case
    predictions = [[1, 1, 0.5, 0.5, 0.2, 0.2, 0.9]]
    references = [[1, 1, 0.1, 0.2, 0.2, 0.2]]
    result = ufm.calculate(predictions, references)

    assert result["tp"] == 0, "No TP expected for non-matching data points"
    assert result["fn"] == 1, "Expected 1 FN for non-matching data points"
    assert result["unique_obj_count"] == 1, "Expected 1 unique GT ID"


def test_calculate_conflicting_ids():
    """
    Tests the calculate function with predictions and references that contain duplicate IDs in the same frame.

    Asserts that only one true positive (TP) is counted, and that the duplicate ID is counted as one false positive (FP).
    """
    predictions = [
        [1, 1, 0.1, 0.1, 0.2, 0.2, 0.9],
        [1, 1, 0.3, 0.3, 0.2, 0.2, 0.8],  # Duplicate ID in same frame
    ]
    references = [[1, 1, 0.1, 0.1, 0.2, 0.2]]

    result = ufm.calculate(predictions, references)

    assert result["tp"] == 1, "Only one TP should be counted for duplicate IDs"
    assert result["fn"] == 0, "No FN expected as the reference is matched"
    assert result["unique_obj_count"] == 1, "Expected 1 unique GT ID"


def test_calculate_mismatched_frames():
    """
    Tests the calculate function with predictions and references that have mismatched frames.

    Asserts that no true positives (TP) are counted, and that all predictions are counted as false positives (FP)
    and all references are counted as false negatives (FN) when the frames do not match.
    """
    predictions = [[1, 1, 0.1, 0.2, 0.3, 0.4, 0.9]]
    references = [[2, 1, 0.1, 0.2, 0.3, 0.4]]  # Different frame

    result = ufm.calculate(predictions, references)

    assert result["tp"] == 0, "No TP expected for mismatched frames"
    assert result["fn"] == 1, "All references should be FN for mismatched frames"
    assert result["unique_obj_count"] == 1, "Expected 1 unique GT ID"


def test_realize_metrics():
    # Test 1: Typical case with valid metrics
    """
    Tests the realize_metrics function.

    Tests the following cases:
    1. Typical case with valid metrics.
    2. Edge case with zero TP, FP, FN.
    3. Zero FP but non-zero TP.
    4. Large unique_obj_count with zero recognized.
    """
    metrics_dict = {
        "tp": 10,
        "fn": 3,
        "unique_obj_count": 8,
        "mostly_tracked_count_0_3": 7,
        "mostly_tracked_count_0_5": 6,
        "mostly_tracked_count_0_8": 4,
    }
    recognition_thresholds = [0.3, 0.5, 0.8]
    result = ufm.realize_metrics(metrics_dict, recognition_thresholds)

    assert math.isclose(
        result["recall"], 10 / (10 + 3), rel_tol=0.00001
    ), f"Expected recall, got {result['recall']}"
    assert math.isclose(
        result["mostly_tracked_score_0_3"], 7 / 8, rel_tol=0.00001
    ), f"Expected mostly_tracked_score_0_3, got {result['mostly_tracked_score_0.3']}"
    assert math.isclose(
        result["mostly_tracked_score_0_5"], 6 / 8, rel_tol=0.00001
    ), f"Expected mostly_tracked_score_0.5, got {result['mostly_tracked_score_0.5']}"
    assert math.isclose(
        result["mostly_tracked_score_0_8"], 4 / 8, rel_tol=0.00001
    ), f"Expected mostly_tracked_score_0_8, got {result['mostly_tracked_score_0.8']}"

    # Test 2: Edge case with zero TP, FP, FN
    metrics_dict = {
        "tp": 0,
        "fn": 0,
        "unique_obj_count": 1,
        "mostly_tracked_count_0_3": 0,
        "mostly_tracked_count_0_5": 0,
        "mostly_tracked_count_0_8": 0,
    }
    recognition_thresholds = [0.3, 0.5, 0.8]
    result = ufm.realize_metrics(metrics_dict, recognition_thresholds)

    assert math.isclose(
        result["recall"], 0, rel_tol=0.00001
    ), f"Expected recall NaN, got {result['recall']}"
    assert (
        result["mostly_tracked_score_0_3"] == 0
    ), f"Expected mostly_tracked_score_0_3, got {result['mostly_tracked_score_0_3']}"
    assert (
        result["mostly_tracked_score_0_5"] == 0
    ), f"Expected mostly_tracked_score_0_5, got {result['mostly_tracked_score_0_5']}"
    assert (
        result["mostly_tracked_score_0_8"] == 0
    ), f"Expected mostly_tracked_score_0_8, got {result['mostly_tracked_score_0_8']}"

    # Test 3: Zero FP but non-zero TP
    metrics_dict = {
        "tp": 5,
        "fn": 2,
        "unique_obj_count": 10,
        "mostly_tracked_count_0_3": 3,
        "mostly_tracked_count_0_5": 1,
    }
    recognition_thresholds = [0.3, 0.5]
    result = ufm.realize_metrics(metrics_dict, recognition_thresholds)

    assert math.isclose(
        result["recall"], 5 / (5 + 2), rel_tol=0.00001
    ), f"Expected recall, got {result['recall']}"
    assert math.isclose(
        result["mostly_tracked_score_0_3"], 3 / 10, rel_tol=0.00001
    ), f"Expected mostly_tracked_score_0.3, got {result['mostly_tracked_score_0_3']}"
    assert math.isclose(
        result["mostly_tracked_score_0_5"], 1 / 10, rel_tol=0.00001
    ), f"Expected mostly_tracked_score_0.5, got {result['mostly_tracked_score_0_5']}"

    # Test 4: Large unique_obj_count with zero recognized
    metrics_dict = {
        "tp": 0,
        "fn": 5,
        "unique_obj_count": 1000,
        "mostly_tracked_count_0_3": 0,
        "mostly_tracked_count_0_5": 0,
    }
    recognition_thresholds = [0.3, 0.5]
    result = ufm.realize_metrics(metrics_dict, recognition_thresholds)

    assert result["recall"] == 0, f"Expected recall 0, got {result['recall']}"
    assert (
        result["mostly_tracked_score_0_3"] == 0
    ), f"Expected mostly_tracked_score_0_3, got {result['mostly_tracked_score_0_3']}"
    assert (
        result["mostly_tracked_score_0_5"] == 0
    ), f"Expected mostly_tracked_score_0_5, got {result['mostly_tracked_score_0_5']}"


def test_unique_obj_count():
    """
    Tests the unique_obj_count function.

    The unique_obj_count function takes a MOTChallenge events DataFrame and returns the number of unique ground truth IDs in the DataFrame.

    The tests cover the following cases:
    1. Typical case with multiple unique IDs
    2. Single frame with one GT ID
    3. Multiple frames, duplicate GT IDs
    4. High number of frames and IDs (stress test)
    5. Same GT ID across all frames
    6. Overlapping and non-overlapping GT IDs

    Each test case runs the unique_obj_count function on a DataFrame created from a MOTChallenge events file and verifies that the output matches the expected number of unique ground truth IDs.

    """

    def run_test_case(
        num_frames,
        np_predictions,
        np_references,
        expected_unique_gt_ids,
        test_case_name,
    ):
        acc = mm.MOTAccumulator(auto_id=True)

        for i in range(1, num_frames + 1):
            preds = np_predictions[np_predictions[:, 0] == i, 1:6]
            refs = np_references[np_references[:, 0] == i, 1:6]
            C = mm.distances.iou_matrix(refs[:, 1:], preds[:, 1:], max_iou=0.5)
            acc.update(
                refs[:, 0].astype("int").tolist(), preds[:, 0].astype("int").tolist(), C
            )

        df = mm.metrics.events_to_df_map(acc.events)
        unique_gt_ids = ufm.unique_obj_count(df)
        assert (
            unique_gt_ids == expected_unique_gt_ids
        ), f"{test_case_name} failed: Expected {expected_unique_gt_ids}, got {unique_gt_ids}"

    # Test 1: Typical case with multiple unique IDs
    run_test_case(
        num_frames=5,
        np_predictions=np.array(
            [
                [1, 101, 0.1, 0.1, 0.2, 0.2],
                [2, 102, 0.1, 0.1, 0.2, 0.2],
                [3, 103, 0.5, 0.5, 0.6, 0.6],
            ]
        ),
        np_references=np.array(
            [
                [1, 201, 0.1, 0.1, 0.2, 0.2],
                [2, 202, 0.1, 0.1, 0.2, 0.2],
                [3, 203, 0.5, 0.5, 0.6, 0.6],
            ]
        ),
        expected_unique_gt_ids=3,
        test_case_name="Test 1: Typical case",
    )

    # Test 2: Single frame with one GT ID
    run_test_case(
        num_frames=1,
        np_predictions=np.array(
            [
                [1, 101, 0.1, 0.1, 0.2, 0.2],
            ]
        ),
        np_references=np.array(
            [
                [1, 201, 0.1, 0.1, 0.2, 0.2],
            ]
        ),
        expected_unique_gt_ids=1,
        test_case_name="Test 2: Single frame with one GT ID",
    )

    # Test 3: Multiple frames, duplicate GT IDs
    run_test_case(
        num_frames=3,
        np_predictions=np.array(
            [
                [1, 101, 0.1, 0.1, 0.2, 0.2],
                [2, 102, 0.3, 0.3, 0.4, 0.4],
            ]
        ),
        np_references=np.array(
            [
                [1, 201, 0.1, 0.1, 0.2, 0.2],
                [2, 201, 0.1, 0.1, 0.2, 0.2],  # Same ID as Frame 1
                [3, 202, 0.5, 0.5, 0.6, 0.6],
            ]
        ),
        expected_unique_gt_ids=2,
        test_case_name="Test 3: Multiple frames, duplicate GT IDs",
    )

    # Test 4: High number of frames and IDs (stress test)
    run_test_case(
        num_frames=100,
        np_predictions=np.array(
            [[i, 100 + i, 0.1, 0.1, 0.2, 0.2] for i in range(1, 101)]
        ),
        np_references=np.array(
            [[i, 200 + i, 0.1, 0.1, 0.2, 0.2] for i in range(1, 101)]
        ),
        expected_unique_gt_ids=100,
        test_case_name="Test 4: High number of frames and IDs (stress test)",
    )

    # Test 5: Same GT ID across all frames
    run_test_case(
        num_frames=5,
        np_predictions=np.array(
            [[i, 100 + i, 0.1, 0.1, 0.2, 0.2] for i in range(1, 6)]
        ),
        np_references=np.array([[i, 201, 0.1, 0.1, 0.2, 0.2] for i in range(1, 6)]),
        expected_unique_gt_ids=1,
        test_case_name="Test 5: Same GT ID across all frames",
    )

    # Test 6: Overlapping and non-overlapping GT IDs
    run_test_case(
        num_frames=4,
        np_predictions=np.array(
            [
                [1, 101, 0.1, 0.1, 0.2, 0.2],
                [2, 102, 0.3, 0.3, 0.4, 0.4],
            ]
        ),
        np_references=np.array(
            [
                [1, 201, 0.1, 0.1, 0.2, 0.2],
                [2, 202, 0.3, 0.3, 0.4, 0.4],
                [3, 203, 0.5, 0.5, 0.6, 0.6],
                [4, 201, 0.7, 0.7, 0.8, 0.8],
            ]
        ),
        expected_unique_gt_ids=3,
        test_case_name="Test 6: Overlapping and non-overlapping GT IDs",
    )