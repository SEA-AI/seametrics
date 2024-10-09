# test_payload_processor.py

import pytest
from unittest.mock import patch, MagicMock
from seametrics.payload.processor import PayloadProcessor

# Mock EXCLUDED_CLASSES if necessary
EXCLUDED_CLASSES = ["class1", "class2"]


@pytest.fixture
def mock_fiftyone():
    """
    Fixture to mock the entire fiftyone (fo) module and its functions.
    """
    # Patch the entire fiftyone module (fo)
    with patch("seametrics.payload.processor.fo") as mock_fo:
        # Set up mock for fo.list_datasets
        mock_fo.list_datasets.return_value = ["valid_dataset"]

        # Set up mock for fo.load_dataset
        mock_dataset = MagicMock()
        mock_dataset.name = "mock_dataset"
        mock_dataset.group_slices = {"rgb", "thermal_wide", "thermal_left"}
        mock_fo.load_dataset.return_value = mock_dataset

        yield mock_fo  # Yield the mock_fo object to be used in the tests


@pytest.fixture
def mock_compute_payload():
    """
    Fixture to mock compute_payload function in the PayloadProcessor class.
    """
    with patch.object(PayloadProcessor, "compute_payload") as mock_compute_payload:
        yield mock_compute_payload


def test_payload_processor_initialization(mock_fiftyone, mock_compute_payload):
    """
    Test that PayloadProcessor initializes correctly with valid inputs.
    """
    # Given
    dataset_name = "valid_dataset"
    gt_field = "ground_truth_det"
    models = ["model1", "model2"]

    # When
    processor = PayloadProcessor(
        dataset_name=dataset_name,
        gt_field=gt_field,
        models=models,
        tracking_mode=False,
        sequence_list=["seq1", "seq2"],
        data_type="thermal",
        excluded_classes=EXCLUDED_CLASSES,
        slices=["slice1", "slice2"],
        tags=["tag1", "tag2"],
    )

    # Then
    assert processor.dataset_name == dataset_name
    assert processor.gt_field == gt_field
    assert processor.models == models
    mock_compute_payload.assert_called_once()


def test_validate_input_parameters_invalid_data_type(
    mock_fiftyone, mock_compute_payload
):
    """
    Test validate_input_parameters with an invalid data_type value.
    """
    with pytest.raises(ValueError, match="data_type must be 'rgb' or 'thermal'"):
        PayloadProcessor(
            dataset_name="valid_dataset",
            gt_field="ground_truth_det",
            models=["model1", "model2"],
            tracking_mode=False,
            data_type="invalid_type",  # Invalid data_type
        )


def test_validate_input_parameters_dataset_not_found(
    mock_fiftyone, mock_compute_payload
):
    """
    Test validate_input_parameters when the dataset is not found in FiftyOne.
    """
    # Mock FiftyOne list_datasets to return an empty list, simulating dataset not found
    mock_fiftyone.list_datasets.return_value = (
        []
    )  # fo.list_datasets returns an empty list

    with pytest.raises(ValueError, match="Dataset valid_dataset not found in FiftyOne"):
        PayloadProcessor(
            dataset_name="valid_dataset",
            gt_field="ground_truth_det",
            models=["model1", "model2"],
            tracking_mode=False,
        )

    mock_fiftyone.list_datasets.assert_called_once()


def test_compute_payload_function_called_once(mock_fiftyone, mock_compute_payload):
    """
    Test that compute_payload is called exactly once during initialization.
    """
    # Given
    dataset_name = "valid_dataset"
    gt_field = "ground_truth_det"
    models = ["model1"]

    # When
    _ = PayloadProcessor(
        dataset_name=dataset_name,
        gt_field=gt_field,
        models=models,
        tracking_mode=False,
    )

    # Then
    mock_compute_payload.assert_called_once()


def test_accidental_tuple_as_models(mock_fiftyone, mock_compute_payload):
    """
    Test that a ValueError is raised if models is a tuple.
    """
    models = (["model1", "model2"],)
    with pytest.raises(TypeError, match="models must be a list"):
        PayloadProcessor(
            dataset_name="valid_dataset",
            gt_field="ground_truth_det",
            models=models,
            tracking_mode=False,
        )
