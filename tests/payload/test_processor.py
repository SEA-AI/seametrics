# test_payload_processor.py

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

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
    mock_fiftyone.list_datasets.return_value = []  # fo.list_datasets returns an empty list

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


def test_payload_initialization_with_frame_ids(mock_fiftyone, mock_compute_payload):
    """
    Test that start_frame_id and end_frame_id are set correctly.
    """
    processor = PayloadProcessor(
        dataset_name="valid_dataset",
        gt_field="ground_truth_det",
        models=["model1"],
        tracking_mode=False,
        start_frame_id=5,
        end_frame_id=10,
    )
    assert processor.start_frame_id == 5
    assert processor.end_frame_id == 11

    processor = PayloadProcessor(
        dataset_name="valid_dataset",
        gt_field="ground_truth_det",
        models=["model1"],
        tracking_mode=False,
        start_frame_id=2,
        end_frame_id=None,
    )
    assert processor.start_frame_id == 2
    assert processor.end_frame_id is None

    processor = PayloadProcessor(
        dataset_name="valid_dataset",
        gt_field="ground_truth_det",
        models=["model1"],
        tracking_mode=False,
    )
    assert processor.start_frame_id is None
    assert processor.end_frame_id is None


# ---------------------------------------------------------------------------
# get_keyframes: record the prediction keyframe flags on the Sequence
# ---------------------------------------------------------------------------


class _FakeView:
    """Minimal stand-in for a video fo.DatasetView.

    Only implements what get_keyframes touches: a media type (so the field path
    can be resolved) and values() lookups on that path.
    """

    media_type = "video"

    def __init__(self, values_by_path=None, raises=False) -> None:
        self._values_by_path = values_by_path or {}
        self._raises = raises

    def values(self, path):
        if self._raises:
            raise ValueError(f"no such field: {path}")
        return self._values_by_path.get(path)


def _processor(**kwargs: object):
    """Build a PayloadProcessor with payload computation stubbed out."""
    defaults = {
        "dataset_name": "valid_dataset",
        "gt_field": "ground_truth_det",
        "models": ["model1"],
        "tracking_mode": False,
    }
    defaults.update(kwargs)
    return PayloadProcessor(**defaults)


KEYFRAME_PATH = "frames[].model1.keyframe"


@pytest.mark.usefixtures("mock_fiftyone", "mock_compute_payload")
class TestGetKeyframes:
    """Reading prediction keyframe flags so they can be stored on the Sequence."""

    def test_returns_flags(self):
        """A prediction field with keyframe data yields one bool per frame."""
        processor = _processor()
        view = _FakeView({KEYFRAME_PATH: [True, False, True, False]})

        assert processor.get_keyframes(view, view, "model1") == [
            True,
            False,
            True,
            False,
        ]

    def test_coerces_truthy_values_to_bool(self):
        """None/0/1 from fiftyone become real bools so downstream masks are clean."""
        processor = _processor()
        view = _FakeView({KEYFRAME_PATH: [1, None, 1, 0]})

        assert processor.get_keyframes(view, view, "model1") == [
            True,
            False,
            True,
            False,
        ]

    def test_returns_none_when_field_missing(self):
        """A plain detection model has no keyframe attribute; that is not an error."""
        processor = _processor()
        view = _FakeView(raises=True)

        assert processor.get_keyframes(view, view, "model1") is None

    def test_returns_none_when_values_are_none(self):
        processor = _processor()
        view = _FakeView({KEYFRAME_PATH: None})

        assert processor.get_keyframes(view, view, "model1") is None

    def test_returns_none_when_no_frame_is_flagged(self):
        """An all-False mask carries no information and must not be recorded."""
        processor = _processor()
        view = _FakeView({KEYFRAME_PATH: [False, False, False]})

        assert processor.get_keyframes(view, view, "model1") is None

    def test_respects_frame_id_range(self):
        """The mask must be sliced like the detections, or it would misalign."""
        processor = _processor(start_frame_id=1, end_frame_id=3)
        view = _FakeView({KEYFRAME_PATH: [True, False, True, True, False]})

        # end_frame_id is stored as end + 1, so frames 1..3 inclusive
        assert processor.get_keyframes(view, view, "model1") == [False, True, True]


# ---------------------------------------------------------------------------
# process_sequence: the keyframe mask must reach the Sequence
# ---------------------------------------------------------------------------


class _FakeSequenceView:
    """Stand-in for the per-sequence view chain used by process_sequence."""

    media_type = "video"

    def __init__(self, values_by_path) -> None:
        self._values_by_path = values_by_path

    def match(self, *_args: object, **_kwargs: object):
        return self

    def filter_labels(self, *_args: object, **_kwargs: object):
        return self

    def first(self):
        return SimpleNamespace(
            metadata=SimpleNamespace(frame_height=50, frame_width=100)
        )

    def values(self, path):
        if path not in self._values_by_path:
            raise ValueError(f"no such field: {path}")
        return self._values_by_path[path]


@pytest.mark.usefixtures("mock_fiftyone", "mock_compute_payload")
class TestProcessSequenceKeyframes:
    """The mask is recorded on the Sequence rather than applied."""

    GT = "ground_truth_det"

    def _run(self, values_by_path, **kwargs: object):
        processor = _processor(**kwargs)
        processor.dataset = _FakeSequenceView(values_by_path)
        return processor.process_sequence("seq-1")

    def test_keyframes_recorded_per_model(self):
        sequence = self._run(
            {
                "frames[].model1.detections": [["d0"], ["d1"], ["d2"]],
                f"frames[].{self.GT}.detections": [["g0"], ["g1"], ["g2"]],
                "frames[].model1.keyframe": [True, False, True],
            }
        )
        assert sequence.keyframes == {"model1": [True, False, True]}

    def test_detections_are_not_blanked_outside_tracking_mode(self):
        """Recording the mask must not change what detections come out."""
        sequence = self._run(
            {
                "frames[].model1.detections": [["d0"], ["d1"], ["d2"]],
                f"frames[].{self.GT}.detections": [["g0"], ["g1"], ["g2"]],
                "frames[].model1.keyframe": [True, False, True],
            }
        )
        assert sequence["model1"] == [["d0"], ["d1"], ["d2"]]
        assert sequence[self.GT] == [["g0"], ["g1"], ["g2"]]

    def test_no_keyframe_entry_for_the_ground_truth_field(self):
        sequence = self._run(
            {
                "frames[].model1.detections": [["d0"], ["d1"]],
                f"frames[].{self.GT}.detections": [["g0"], ["g1"]],
                "frames[].model1.keyframe": [True, False],
                f"frames[].{self.GT}.keyframe": [True, True],
            }
        )
        assert self.GT not in sequence.keyframes

    def test_keyframes_empty_when_model_has_no_flags(self):
        """A plain detection model yields an empty mapping, not a crash."""
        sequence = self._run(
            {
                "frames[].model1.detections": [["d0"], ["d1"]],
                f"frames[].{self.GT}.detections": [["g0"], ["g1"]],
            }
        )
        assert sequence.keyframes == {}

    def test_keyframes_recorded_for_every_model(self):
        sequence = self._run(
            {
                "frames[].model1.detections": [["a0"], ["a1"]],
                "frames[].model2.detections": [["b0"], ["b1"]],
                f"frames[].{self.GT}.detections": [["g0"], ["g1"]],
                "frames[].model1.keyframe": [True, False],
                "frames[].model2.keyframe": [False, True],
            },
            models=["model1", "model2"],
        )
        assert sequence.keyframes == {
            "model1": [True, False],
            "model2": [False, True],
        }

    def test_tracking_mode_still_blanks_non_keyframes(self):
        """The pre-existing blanking behaviour is unchanged."""
        sequence = self._run(
            {
                "frames[].model1.detections": [["d0"], ["d1"], ["d2"]],
                f"frames[].{self.GT}.detections": [["g0"], ["g1"], ["g2"]],
                "frames[].model1.keyframe": [True, False, True],
            },
            tracking_mode=True,
        )
        assert sequence["model1"] == [["d0"], [], ["d2"]]
        assert sequence[self.GT] == [["g0"], [], ["g2"]]
        assert sequence.keyframes == {"model1": [True, False, True]}

    def test_keyframes_excluded_from_field_names(self):
        sequence = self._run(
            {
                "frames[].model1.detections": [["d0"]],
                f"frames[].{self.GT}.detections": [["g0"]],
                "frames[].model1.keyframe": [True],
            }
        )
        assert sorted(sequence.field_names) == sorted([self.GT, "model1"])
