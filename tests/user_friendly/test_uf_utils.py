import math

import motmetrics as mm
import numpy as np
import pandas as pd
import pytest

from seametrics.payload import Payload, Resolution, Sequence
from seametrics.user_friendly.utils import (
    build_metrics_template,
    calculate,
    calculate_from_payload,
    num_gt_ids,
    realize_metrics,
    recognition,
    sum_dicts,
)


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
    assert recognition(track_ratios, th=th) == 4

    th = 0.5
    assert recognition(track_ratios, th=th) == 3

    th = 0.8
    assert recognition(track_ratios, th=th) == 2

    th = 1.0
    assert recognition(track_ratios, th=th) == 1

    th = 1.1
    assert recognition(track_ratios, th=th) == 0


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

    result = calculate(predictions, references)

    assert "tp" in result
    assert "fp" in result
    assert "fn" in result

    assert result["tp"] == 2.0, f"Expected tp to be 2.0, got {result['tp']}"
    assert result["fp"] == 0.0, f"Expected fp to be 0.0, got {result['fp']}"
    assert result["fn"] == 0.0, f"Expected fn to be 0.0, got {result['fn']}"
    assert (
        result["num_gt_ids"] == 2
    ), f"Expected num_gt_ids to be 2, got {result['num_gt_ids']}"
    assert (
        result["recognized_0.3"] == 2
    ), f"Expected recognized_0.3 to be 2, got {result['recognized_0.3']}"
    assert (
        result["recognized_0.5"] == 2
    ), f"Expected recognized_0.5 to be 2, got {result['recognized_0.5']}"
    assert (
        result["recognized_0.8"] == 2
    ), f"Expected recognized_0.8 to be 2, got {result['recognized_0.8']}"


def test_calculate_empty_inputs_with_exceptions():
    """
    Tests the calculate function for cases with empty inputs.

    Ensures that ValueError is raised when:
    - Both predictions and references are empty.
    - Predictions are empty but references are non-empty.
    - References are empty but predictions are non-empty.

    Checks that the appropriate error message is provided for each case.
    """
    predictions = []
    references = []

    # Both empty
    with pytest.raises(
        ValueError,
        match="The predictions should be a 2D array with 7 columns",
    ):
        calculate(predictions, references)

    # Empty predictions, non-empty references
    references = [[1, 1, 0.1, 0.2, 0.3, 0.4]]
    with pytest.raises(
        ValueError,
        match="The predictions should be a 2D array with 7 columns",
    ):
        calculate(predictions, references)

    # Empty references, non-empty predictions
    predictions = [[1, 1, 0.1, 0.2, 0.3, 0.4, 0.9]]
    references = []
    with pytest.raises(
        ValueError,
        match="The references should be a 2D array with 6 columns",
    ):
        calculate(predictions, references)


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
        calculate(predictions, references)

    # Invalid shape for references (missing width and height)
    predictions = [[1, 1, 0.1, 0.2, 0.3, 0.4, 0.9]]
    references = [[1, 1, 0.1, 0.2]]  # Only 4 columns instead of 6

    with pytest.raises(
        ValueError,
        match="The references should be a 2D array with 6 columns",
    ):
        calculate(predictions, references)


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
        calculate(predictions, references)

    # Invalid frame number in references (frame number is negative)
    predictions = [[1, 1, 0.1, 0.2, 0.3, 0.4, 0.9]]
    references = [[-1, 1, 0.1, 0.2, 0.3, 0.4]]

    with pytest.raises(
        ValueError,
        match="The frame number in the references should be a positive integer",
    ):
        calculate(predictions, references)


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
    result = calculate(predictions, references)

    assert result["tp"] == 1, "Expected 1 TP for a matching single data point"
    assert result["fp"] == 0, "No FP expected for a matching single data point"
    assert result["fn"] == 0, "No FN expected for a matching single data point"
    assert result["num_gt_ids"] == 1, "Expected 1 unique GT ID"

    # Non-matching case
    predictions = [[1, 1, 0.5, 0.5, 0.2, 0.2, 0.9]]
    references = [[1, 1, 0.1, 0.2, 0.2, 0.2]]
    result = calculate(predictions, references)

    assert result["tp"] == 0, "No TP expected for non-matching data points"
    assert result["fp"] == 1, "Expected 1 FP for non-matching data points"
    assert result["fn"] == 1, "Expected 1 FN for non-matching data points"
    assert result["num_gt_ids"] == 1, "Expected 1 unique GT ID"


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

    result = calculate(predictions, references)

    assert result["tp"] == 1, "Only one TP should be counted for duplicate IDs"
    assert result["fp"] == 1, "One FP expected due to duplicate ID"
    assert result["fn"] == 0, "No FN expected as the reference is matched"
    assert result["num_gt_ids"] == 1, "Expected 1 unique GT ID"


def test_calculate_mismatched_frames():
    """
    Tests the calculate function with predictions and references that have mismatched frames.

    Asserts that no true positives (TP) are counted, and that all predictions are counted as false positives (FP)
    and all references are counted as false negatives (FN) when the frames do not match.
    """
    predictions = [[1, 1, 0.1, 0.2, 0.3, 0.4, 0.9]]
    references = [[2, 1, 0.1, 0.2, 0.3, 0.4]]  # Different frame

    result = calculate(predictions, references)

    assert result["tp"] == 0, "No TP expected for mismatched frames"
    assert result["fp"] == 1, "All predictions should be FP for mismatched frames"
    assert result["fn"] == 1, "All references should be FN for mismatched frames"
    assert result["num_gt_ids"] == 1, "Expected 1 unique GT ID"


def test_calculate_empty_predictions_or_references():
    # Empty predictions
    """
    Tests the calculate function with empty predictions or references.

    Ensures that ValueError is raised when:
    - Predictions are empty.
    - References are empty.

    Checks that the appropriate error message is provided for each case.
    """
    predictions = []
    references = [[1, 1, 0.1, 0.2, 0.3, 0.4]]

    with pytest.raises(
        ValueError,
        match="The predictions should be a 2D array with 7 columns",
    ):
        calculate(predictions, references)

    # Empty references
    predictions = [[1, 1, 0.1, 0.2, 0.3, 0.4, 0.9]]
    references = []

    with pytest.raises(
        ValueError,
        match="The references should be a 2D array with 6 columns",
    ):
        calculate(predictions, references)


def test_calculate_none_inputs():
    """
    Tests the calculate function with None inputs.

    Ensures that ValueError is raised when:
    - Predictions are None.
    - References are None.

    Checks that the appropriate error message is provided for each case.
    """
    predictions = None
    references = [[1, 1, 0.1, 0.2, 0.3, 0.4]]

    # Test None predictions
    with pytest.raises(
        ValueError,
        match="The predictions should be a 2D array with 7 columns",
    ):
        calculate(predictions, references)

    predictions = [[1, 1, 0.1, 0.2, 0.3, 0.4, 0.9]]
    references = None

    # Test None references
    with pytest.raises(
        ValueError,
        match="The references should be a 2D array with 6 columns",
    ):
        calculate(predictions, references)


def test_sum_dicts():
    # Test 1: Simple dictionaries with overlapping keys
    """
    Tests the sum_dicts function.

    Tests the following cases:
    1. Simple dictionaries with overlapping keys.
    2. Missing keys in one dictionary.
    3. Non-overlapping keys.
    4. Empty dictionaries.
    5. One empty dictionary.
    6. Non-numerical values (should take the non-zero value).
    """
    dict1 = {"a": 1, "b": 2, "c": {"d": 4, "e": 5}}
    dict2 = {"a": 3, "b": 4, "c": {"d": 6, "f": 7}, "g": 8}
    expected = {"a": 4, "b": 6, "c": {"d": 10, "e": 5, "f": 7}, "g": 8}
    result = sum_dicts(dict1, dict2)
    assert result == expected, f"Test 1 failed: {result}"

    # Test 2: Missing keys in one dictionary
    dict1 = {"a": 1, "b": {"c": 2}}
    dict2 = {"b": {"d": 3}, "e": 4}
    expected = {"a": 1, "b": {"c": 2, "d": 3}, "e": 4}
    result = sum_dicts(dict1, dict2)
    assert result == expected, f"Test 2 failed: {result}"

    # Test 3: Non-overlapping keys
    dict1 = {"x": 1, "y": {"z": 2}}
    dict2 = {"a": 3, "b": {"c": 4}}
    expected = {"x": 1, "y": {"z": 2}, "a": 3, "b": {"c": 4}}
    result = sum_dicts(dict1, dict2)
    assert result == expected, f"Test 3 failed: {result}"

    # Test 4: Empty dictionaries
    dict1 = {}
    dict2 = {}
    expected = {}
    result = sum_dicts(dict1, dict2)
    assert result == expected, f"Test 4 failed: {result}"

    # Test 5: One empty dictionary
    dict1 = {"a": 1, "b": {"c": 2}}
    dict2 = {}
    expected = {"a": 1, "b": {"c": 2}}
    result = sum_dicts(dict1, dict2)
    assert result == expected, f"Test 5 failed: {result}"

    # Test 6: Non-numerical values (should take the non-zero value)
    dict1 = {"a": "text1", "b": 2}
    dict2 = {"a": "text2", "b": 3}
    expected = {"a": "text1", "b": 5}
    result = sum_dicts(dict1, dict2)
    assert result == expected, f"Test 6 failed: {result}"


def test_build_metrics_template():
    # Test 1: Typical case with multiple models and filters
    """
    Tests the build_metrics_template function.

    Tests the following cases:
    1. Typical case with multiple models and filters.
    2. Empty models list.
    3. Empty filters dictionary.
    4. Single model and single filter with no ranges.
    5. Large number of models and filters.
    """
    models = ["model1", "model2"]
    filters = {
        "filter1": [("range1", 1), ("range2", 2)],
        "filter2": [("range3", 3)],
    }
    expected = {
        "model1": {
            "all": {},
            "filter1": {
                "range1": {},
                "range2": {},
            },
            "filter2": {
                "range3": {},
            },
        },
        "model2": {
            "all": {},
            "filter1": {
                "range1": {},
                "range2": {},
            },
            "filter2": {
                "range3": {},
            },
        },
    }
    result = build_metrics_template(models, filters)
    assert result == expected, f"Test 1 failed: {result}"

    # Test 2: Empty models list
    models = []
    filters = {
        "filter1": [("range1", 1)],
    }
    expected = {}
    result = build_metrics_template(models, filters)
    assert result == expected, f"Test 2 failed: {result}"

    # Test 3: Empty filters dictionary
    models = ["model1"]
    filters = {}
    expected = {
        "model1": {
            "all": {},
        },
    }
    result = build_metrics_template(models, filters)
    assert result == expected, f"Test 3 failed: {result}"

    # Test 4: Single model and single filter with no ranges
    models = ["model1"]
    filters = {"filter1": []}
    expected = {
        "model1": {
            "all": {},
            "filter1": {},
        },
    }
    result = build_metrics_template(models, filters)
    assert result == expected, f"Test 4 failed: {result}"

    # Test 5: Large number of models and filters
    models = [f"model{i}" for i in range(5)]
    filters = {f"filter{j}": [(f"range{k}", k) for k in range(3)] for j in range(3)}
    result = build_metrics_template(models, filters)
    for model in models:
        assert model in result, f"Model {model} missing in result"
        assert "all" in result[model], f"'all' missing for model {model}"
        for filter_name, filter_ranges in filters.items():
            assert (
                filter_name in result[model]
            ), f"Filter {filter_name} missing for model {model}"
            for filter_range in filter_ranges:
                assert (
                    filter_range[0] in result[model][filter_name]
                ), f"Range {filter_range[0]} missing for filter {filter_name} in model {model}"


def test_realize_metrics():
    # Test 1: Typical case with valid metrics
    """
    Tests the realize_metrics function.

    Tests the following cases:
    1. Typical case with valid metrics.
    2. Edge case with zero TP, FP, FN.
    3. Zero FP but non-zero TP.
    4. Large num_gt_ids with zero recognized.
    """
    metrics_dict = {
        "tp": 10,
        "fp": 5,
        "fn": 3,
        "num_gt_ids": 8,
        "recognized_0.3": 7,
        "recognized_0.5": 6,
        "recognized_0.8": 4,
    }
    recognition_thresholds = [0.3, 0.5, 0.8]
    result = realize_metrics(metrics_dict, recognition_thresholds)

    assert result["precision"] == 10 / (
        10 + 5
    ), f"Expected precision, got {result['precision']}"
    assert result["recall"] == 10 / (10 + 3), f"Expected recall, got {result['recall']}"
    assert result["f1"] == (
        2
        * result["precision"]
        * result["recall"]
        / (result["precision"] + result["recall"] + 1e-6)
    ), f"Expected f1, got {result['f1']}"
    assert (
        result["recognition_0.3"] == 7 / 8
    ), f"Expected recognition_0.3, got {result['recognition_0.3']}"
    assert (
        result["recognition_0.5"] == 6 / 8
    ), f"Expected recognition_0.5, got {result['recognition_0.5']}"
    assert (
        result["recognition_0.8"] == 4 / 8
    ), f"Expected recognition_0.8, got {result['recognition_0.8']}"

    # Test 2: Edge case with zero TP, FP, FN
    metrics_dict = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "num_gt_ids": 1,
        "recognized_0.3": 0,
        "recognized_0.5": 0,
        "recognized_0.8": 0,
    }
    recognition_thresholds = [0.3, 0.5, 0.8]
    result = realize_metrics(metrics_dict, recognition_thresholds)

    assert np.isnan(
        result["precision"]
    ), f"Expected precision NaN, got {result['precision']}"
    assert np.isnan(result["recall"]), f"Expected recall NaN, got {result['recall']}"
    assert np.isnan(result["f1"]), f"Expected f1 NaN, got {result['f1']}"
    assert (
        result["recognition_0.3"] == 0
    ), f"Expected recognition_0.3, got {result['recognition_0.3']}"
    assert (
        result["recognition_0.5"] == 0
    ), f"Expected recognition_0.5, got {result['recognition_0.5']}"
    assert (
        result["recognition_0.8"] == 0
    ), f"Expected recognition_0.8, got {result['recognition_0.8']}"

    # Test 3: Zero FP but non-zero TP
    metrics_dict = {
        "tp": 5,
        "fp": 0,
        "fn": 2,
        "num_gt_ids": 10,
        "recognized_0.3": 3,
        "recognized_0.5": 1,
    }
    recognition_thresholds = [0.3, 0.5]
    result = realize_metrics(metrics_dict, recognition_thresholds)

    assert result["precision"] == 5 / (
        5 + 0
    ), f"Expected precision, got {result['precision']}"
    assert result["recall"] == 5 / (5 + 2), f"Expected recall, got {result['recall']}"
    assert result["f1"] == (
        2
        * result["precision"]
        * result["recall"]
        / (result["precision"] + result["recall"] + 1e-6)
    ), f"Expected f1, got {result['f1']}"
    assert (
        result["recognition_0.3"] == 3 / 10
    ), f"Expected recognition_0.3, got {result['recognition_0.3']}"
    assert (
        result["recognition_0.5"] == 1 / 10
    ), f"Expected recognition_0.5, got {result['recognition_0.5']}"

    # Test 4: Large num_gt_ids with zero recognized
    metrics_dict = {
        "tp": 0,
        "fp": 0,
        "fn": 5,
        "num_gt_ids": 1000,
        "recognized_0.3": 0,
        "recognized_0.5": 0,
    }
    recognition_thresholds = [0.3, 0.5]
    result = realize_metrics(metrics_dict, recognition_thresholds)

    assert np.isnan(
        result["precision"]
    ), f"Expected precision NaN, got {result['precision']}"
    assert result["recall"] == 0, f"Expected recall 0, got {result['recall']}"
    assert np.isnan(result["f1"]), f"Expected f1 NaN, got {result['f1']}"
    assert (
        result["recognition_0.3"] == 0
    ), f"Expected recognition_0.3, got {result['recognition_0.3']}"
    assert (
        result["recognition_0.5"] == 0
    ), f"Expected recognition_0.5, got {result['recognition_0.5']}"


def test_num_gt_ids():
    """
    Tests the num_gt_ids function.

    The num_gt_ids function takes a MOTChallenge events DataFrame and returns the number of unique ground truth IDs in the DataFrame.

    The tests cover the following cases:
    1. Typical case with multiple unique IDs
    2. Single frame with one GT ID
    3. Multiple frames, duplicate GT IDs
    4. High number of frames and IDs (stress test)
    5. Same GT ID across all frames
    6. Overlapping and non-overlapping GT IDs

    Each test case runs the num_gt_ids function on a DataFrame created from a MOTChallenge events file and verifies that the output matches the expected number of unique ground truth IDs.

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
        unique_gt_ids = num_gt_ids(df)
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


def test_calculate_from_payload():
    """
    Tests the calculate_from_payload function with a single sequence and a few detections.

    The test payload contains a single sequence with two frames. The first frame has one ground
    truth detection and two model detections. The second frame has one ground truth detection and
    one model detection.

    The function is called with max_iou=0.5 and recognition_thresholds=[0.3, 0.5, 0.8]. The test
    asserts that the output contains the correct values for false positives, false negatives,
    true positives, precision, recall, F1 score, recognition at different thresholds, and the
    number of ground truth IDs. The test also asserts that these values are correct for both the
    global metrics and the per-sequence metrics.
    """

    def create_test_payload():
        """
        Creates a test Payload object with a single sequence and a few detections.

        Returns:
            Payload: A test Payload object.
        """
        gt_config = [1, 1]
        model_config = [2, 1]

        mock_detections = [
            {
                "id": "674f2c83d608ab75c380194c",
                "attributes": {},
                "tags": [],
                "label": "MOTORBOAT",
                "bounding_box": [0.0, 0.0, 0.015625, 0.009765625],
                "mask": None,
                "confidence": None,
                "index": 1,
            },
            {
                "id": "6682d49a4cb7459c1be09c52",
                "attributes": {},
                "tags": [],
                "label": "SPHERICAL_BUOY",
                "bounding_box": [0.603125, 0.591796875, 0.0109375, 0.009765625],
                "mask": None,
                "confidence": None,
                "index": 2,
            },
        ]

        payload = Payload(
            dataset="dataset",
            models=["model"],
            gt_field_name="ground_truth_det",
            sequences={
                "sequence_a": Sequence(
                    resolution=Resolution(height=512, width=640),
                    ground_truth_det=[
                        [mock_detections[i] for i in range(detections)]
                        for detections in gt_config
                    ],
                    model=[
                        [mock_detections[i] for i in range(detections)]
                        for detections in model_config
                    ],
                )
            },
        )

        return payload

    payload = create_test_payload()
    max_iou = 0.5
    recognition_thresholds = [0.3, 0.5, 0.8]
    debug = False

    output = calculate_from_payload(
        payload=payload,
        max_iou=max_iou,
        recognition_thresholds=recognition_thresholds,
        debug=debug,
    )

    global_metrics = output["global"]["model"]["all"]
    assert global_metrics["fp"] == 1.0, "False positives mismatch"
    assert global_metrics["fn"] == 0.0, "False negatives mismatch"
    assert global_metrics["tp"] == 2.0, "True positives mismatch"
    assert global_metrics["num_gt_ids"] == 1, "Number of ground truth IDs mismatch"
    assert math.isclose(
        global_metrics["precision"], 0.6666666666666666, rel_tol=1e-9
    ), "Precision mismatch"
    assert global_metrics["recall"] == 1.0, "Recall mismatch"
    assert math.isclose(
        global_metrics["f1"], 0.7999995200002881, rel_tol=1e-9
    ), "F1 score mismatch"
    assert global_metrics["recognition_0.3"] == 1.0, "Recognition at 0.3 mismatch"
    assert global_metrics["recognition_0.5"] == 1.0, "Recognition at 0.5 mismatch"
    assert global_metrics["recognition_0.8"] == 1.0, "Recognition at 0.8 mismatch"
    assert global_metrics["recognized_0.3"] == 1, "Recognized count at 0.3 mismatch"
    assert global_metrics["recognized_0.5"] == 1, "Recognized count at 0.5 mismatch"
    assert global_metrics["recognized_0.8"] == 1, "Recognized count at 0.8 mismatch"

    sequence_metrics = output["per_sequence"]["sequence_a"]["model"]["all"]
    assert sequence_metrics["fp"] == 1.0, "Per-sequence false positives mismatch"
    assert sequence_metrics["fn"] == 0.0, "Per-sequence false negatives mismatch"
    assert sequence_metrics["tp"] == 2.0, "Per-sequence true positives mismatch"
    assert sequence_metrics["num_gt_ids"] == 1, "Per-sequence ground truth IDs mismatch"
    assert math.isclose(
        global_metrics["precision"], 0.6666666666666666, rel_tol=1e-9
    ), "Per-sequence Precision mismatch"
    assert sequence_metrics["recall"] == 1.0, "Per-sequence recall mismatch"
    assert math.isclose(
        global_metrics["f1"], 0.7999995200002881, rel_tol=1e-9
    ), "Per-sequence F1 score mismatch"
    assert (
        sequence_metrics["recognition_0.3"] == 1.0
    ), "Per-sequence recognition at 0.3 mismatch"
    assert (
        sequence_metrics["recognition_0.5"] == 1.0
    ), "Per-sequence recognition at 0.5 mismatch"
    assert (
        sequence_metrics["recognition_0.8"] == 1.0
    ), "Per-sequence recognition at 0.8 mismatch"
    assert (
        sequence_metrics["recognized_0.3"] == 1
    ), "Per-sequence recognized count at 0.3 mismatch"
    assert (
        sequence_metrics["recognized_0.5"] == 1
    ), "Per-sequence recognized count at 0.5 mismatch"
    assert (
        sequence_metrics["recognized_0.8"] == 1
    ), "Per-sequence recognized count at 0.8 mismatch"
