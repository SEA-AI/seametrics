# test_payload_processor.py

from unittest.mock import MagicMock, patch

import fiftyone as fo
import pytest
from pymongo.errors import OperationFailure

from seametrics.payload import Resolution
from seametrics.payload.processor import PayloadProcessor

# Mock EXCLUDED_CLASSES if necessary
EXCLUDED_CLASSES = ["class1", "class2"]
GT = "ground_truth_det"
# One unwound sequence: frame 1 has no ground truth and no keyframe flag
FRAMES = {
    "frames[].model1.detections": [["d0"], ["d1"], ["d2"]],
    f"frames[].{GT}.detections": [["g0"], None, ["g2"]],
    "frames[].model1.keyframe": [True, None, True],
}


@pytest.fixture
def mock_fiftyone():
    """Fixture to mock the entire fiftyone (fo) module and its functions.

    Yields:
        MagicMock: The patched ``fo`` module.
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
    """Fixture to mock compute_payload function in the PayloadProcessor class.

    Yields:
        MagicMock: The patched ``compute_payload`` method.
    """
    with patch.object(PayloadProcessor, "compute_payload") as mock_compute_payload:
        yield mock_compute_payload


@pytest.mark.usefixtures("mock_fiftyone")
def test_payload_processor_initialization(mock_compute_payload):
    """Test that PayloadProcessor initializes correctly with valid inputs."""
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


@pytest.mark.usefixtures("mock_fiftyone", "mock_compute_payload")
def test_validate_input_parameters_invalid_data_type():
    """Test validate_input_parameters with an invalid data_type value."""
    with pytest.raises(ValueError, match="data_type must be 'rgb' or 'thermal'"):
        PayloadProcessor(
            dataset_name="valid_dataset",
            gt_field="ground_truth_det",
            models=["model1", "model2"],
            tracking_mode=False,
            data_type="invalid_type",  # Invalid data_type
        )


@pytest.mark.usefixtures("mock_compute_payload")
def test_validate_input_parameters_dataset_not_found(mock_fiftyone):
    """Test validate_input_parameters when the dataset is not found in FiftyOne."""
    # Mock FiftyOne list_datasets to return an empty list, simulating dataset not found
    mock_fiftyone.list_datasets.return_value = []

    with pytest.raises(ValueError, match="Dataset valid_dataset not found in FiftyOne"):
        PayloadProcessor(
            dataset_name="valid_dataset",
            gt_field="ground_truth_det",
            models=["model1", "model2"],
            tracking_mode=False,
        )

    mock_fiftyone.list_datasets.assert_called_once()


@pytest.mark.usefixtures("mock_fiftyone", "mock_compute_payload")
def test_accidental_tuple_as_models():
    """Test that a TypeError is raised if models is a tuple."""
    models = (["model1", "model2"],)
    with pytest.raises(TypeError, match="models must be a list"):
        PayloadProcessor(
            dataset_name="valid_dataset",
            gt_field="ground_truth_det",
            models=models,
            tracking_mode=False,
        )


@pytest.mark.usefixtures("mock_fiftyone", "mock_compute_payload")
def test_payload_initialization_with_frame_ids():
    """Test that start_frame_id and end_frame_id are set correctly."""
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


def _fetch(view, names, **kwargs: object):
    """Fetch *names* from *view* through a stubbed PayloadProcessor."""
    processor = _processor(**kwargs)
    processor.dataset = view
    return processor.fetch_sequences(names)


# ---------------------------------------------------------------------------
# process_sequence: the keyframe mask must reach the Sequence
# ---------------------------------------------------------------------------


class _FakeSequenceView:
    """Stand-in for the batched view chain used by fetch_sequences."""

    media_type = "video"

    def __init__(self, values_by_path, _stats=None) -> None:
        self._values_by_path = values_by_path
        self._stats = _stats if _stats is not None else {"queries": 0}

    @property
    def query_count(self):
        return self._stats["queries"]

    def match(self, expression=None, *_args: object, **_kwargs: object):
        """Honour `F("sequence").is_in([...])` so batch splitting really narrows."""
        if expression is None:
            return self
        wanted = expression.to_mongo()["$in"][1]
        names = self._values_by_path["sequence"]
        keep = [i for i, name in enumerate(names) if name in wanted]
        # unwound paths hold one entry per frame, not per sample, so match leaves them
        narrowed = {
            path: (
                [value[i] for i in keep]
                if isinstance(value, list) and not path.startswith("frames[].")
                else value
            )
            for path, value in self._values_by_path.items()
        }
        return type(self)(narrowed, self._stats)

    def filter_labels(self, *_args: object, **_kwargs: object):
        return self

    def values(self, path):
        """One query; a list of paths gives one aligned list per path."""
        self._stats["queries"] += 1
        paths = list(path) if isinstance(path, (list, tuple)) else [path]
        found = [self._value(p) for p in paths]
        return found if isinstance(path, (list, tuple)) else found[0]

    def _value(self, path):
        if path.endswith(".keyframe") and path not in self._values_by_path:
            return [None] * len(self._values_by_path["id"])  # as fiftyone does
        if path not in self._values_by_path:
            raise ValueError(f"no such field: {path}")
        return self._values_by_path[path]


class _NestedTooLargeView(_FakeSequenceView):
    """Rejects every nested frame query as oversized; unwound queries fit."""

    oversized = "frames."

    def _value(self, path):
        if path.startswith(self.oversized):
            raise OperationFailure(
                "Executor error during getMore :: caused by :: BSONObj size:"
                " 17355147 is invalid. Size must be between 0 and 16793600(16MB)"
            )
        return super()._value(path)


class _TooLargeView(_NestedTooLargeView):
    """Rejects unwound frame queries as oversized too."""

    oversized = "frames"


class _ReorderedFieldsView(_FakeSequenceView):
    """Serves label-field queries with the samples in reverse order.

    MongoDB does not promise one sample order across separate queries, so a
    field's values must be matched to their sample by id, not by position.
    """

    def filter_labels(self, *_args: object, **_kwargs: object):
        flipped = {
            path: (
                value[::-1]
                if isinstance(value, list) and not path.startswith("frames[].")
                else value
            )
            for path, value in self._values_by_path.items()
        }
        return _FakeSequenceView(flipped, self._stats)


class _ReorderedImageView(_ReorderedFieldsView):
    """Image dataset whose label-field queries come back in reverse sample order."""

    media_type = "image"


def _nest(values_by_path: dict) -> dict:
    """Convert an unwound single-sequence fixture to the nested batch shape."""
    nested = {
        "id": ["seq-1-id"],
        "sequence": ["seq-1"],
        "metadata.frame_width": [100],
        "metadata.frame_height": [50],
    }
    for path, value in values_by_path.items():
        nested[path.replace("frames[].", "frames.")] = [value]
    return nested


@pytest.mark.usefixtures("mock_fiftyone", "mock_compute_payload")
@pytest.mark.parametrize(
    ("tracking_mode", "model", "gt"),
    [
        (False, [["d0"], ["d1"], ["d2"]], [["g0"], [], ["g2"]]),
        (True, [["d0"], [], ["d2"]], [["g0"], [], ["g2"]]),
    ],
)
def test_keyframe_mask_blanks_detections_only_in_tracking_mode(
    tracking_mode, model, gt
):
    processor = _processor(tracking_mode=tracking_mode)
    processor.dataset = _FakeSequenceView(_nest(FRAMES))
    sequence = processor.process_sequence("seq-1")
    assert sequence["model1"] == model
    assert sequence[GT] == gt
    assert sequence.keyframes == {"model1": [True, False, True]}
    assert sorted(sequence.field_names) == sorted([GT, "model1"])


@pytest.mark.usefixtures("mock_fiftyone", "mock_compute_payload")
class TestBatchedFetching:
    """One query per field per batch, with a fallback for oversized results."""

    def _view(self, n_sequences, view_class=_FakeSequenceView):
        names = [f"seq-{i}" for i in range(n_sequences)]
        paths = {
            "id": [f"{name}-id" for name in names],
            "sequence": names,
            "metadata.frame_width": [100] * n_sequences,
            "metadata.frame_height": [50] * n_sequences,
            "frames.model1.detections": [[["d"]] for _ in names],
            f"frames.{GT}.detections": [[["g"]] for _ in names],
            "frames.model1.keyframe": [[True] for _ in names],
        }
        return names, view_class(paths)

    def test_frame_range_slices_every_field(self):
        view = _FakeSequenceView(_nest(FRAMES))
        sequence = _fetch(view, ["seq-1"], start_frame_id=1, end_frame_id=3)["seq-1"]
        assert sequence["model1"] == [["d1"], ["d2"]]
        assert sequence[GT] == [[], ["g2"]]
        assert sequence.keyframes == {"model1": [False, True]}

    def test_a_window_without_keyframes_is_blanked_in_tracking_mode(self):
        """Frame 1 has no flag, but the model flags keyframes elsewhere."""
        view = _FakeSequenceView(_nest(FRAMES))
        sequence = _fetch(
            view, ["seq-1"], start_frame_id=1, end_frame_id=1, tracking_mode=True
        )["seq-1"]
        assert sequence["model1"] == [[]]
        assert sequence.keyframes == {"model1": [False]}

    def test_single_sequence_too_large_still_raises(self):
        """A sequence too large even when unwound raises."""
        names, view = self._view(1, _TooLargeView)
        with pytest.raises(OperationFailure, match="BSONObj size"):
            _fetch(view, names)

    def test_single_oversized_sequence_is_refetched_unwound(self):
        """Unwinding streams frames as separate documents, so the sequence fits."""
        view = _NestedTooLargeView(
            {
                "id": ["s1"],
                "sequence": ["seq-1"],
                "metadata.frame_width": [100],
                "metadata.frame_height": [50],
                **FRAMES,
            }
        )

        sequence = _fetch(view, ["seq-1"])["seq-1"]

        assert sequence["model1"] == [["d0"], ["d1"], ["d2"]]
        assert sequence[GT] == [["g0"], [], ["g2"]]
        assert sequence.keyframes == {"model1": [True, False, True]}

    def test_image_samples_are_grouped_per_sequence_in_view_order(self):
        """Interleaved per-frame samples, read back in another order per field."""
        view = _ReorderedImageView(
            {
                "id": ["i0", "i1", "i2", "i3"],
                "sequence": ["seq-a", "seq-b", "seq-a", "seq-b"],
                "metadata.width": [640, 320, 640, 320],
                "metadata.height": [480, 240, 480, 240],
                "model1.detections": [["a0"], ["b0"], ["a1"], None],
                f"{GT}.detections": [["ga0"], ["gb0"], None, ["gb1"]],
                "model1.keyframe": [True, False, True, True],
            }
        )

        sequences = _fetch(view, ["seq-a", "seq-b"])

        assert sequences["seq-a"]["model1"] == [["a0"], ["a1"]]
        assert sequences["seq-a"][GT] == [["ga0"], []]
        assert sequences["seq-a"].keyframes == {"model1": [True, True]}
        assert sequences["seq-a"].resolution == Resolution(height=480, width=640)
        assert sequences["seq-b"]["model1"] == [["b0"], []]
        assert sequences["seq-b"][GT] == [["gb0"], ["gb1"]]
        assert sequences["seq-b"].keyframes == {"model1": [False, True]}
        assert sequences["seq-b"].resolution == Resolution(height=240, width=320)

    def test_video_fields_follow_their_own_sample(self):
        """A field queried in another sample order still lands on its sequence."""
        view = _ReorderedFieldsView(
            {
                "id": ["s1", "s2"],
                "sequence": ["seq-a", "seq-b"],
                "metadata.frame_width": [100, 200],
                "metadata.frame_height": [50, 80],
                "frames.model1.detections": [[["a0"]], [["b0"], ["b1"]]],
                f"frames.{GT}.detections": [[["ga0"]], [["gb0"], ["gb1"]]],
                "frames.model1.keyframe": [[True], [False, True]],
            }
        )

        sequences = _fetch(view, ["seq-a", "seq-b"])

        assert sequences["seq-a"]["model1"] == [["a0"]]
        assert sequences["seq-a"][GT] == [["ga0"]]
        assert sequences["seq-b"]["model1"] == [["b0"], ["b1"]]
        assert sequences["seq-b"].keyframes == {"model1": [False, True]}
        assert sequences["seq-b"].resolution == Resolution(height=80, width=200)

    def test_models_without_flags_get_no_keyframe_mask(self):
        view = _FakeSequenceView(
            {
                "id": ["s1"],
                "sequence": ["seq-a"],
                "metadata.frame_width": [100],
                "metadata.frame_height": [50],
                "frames.model1.detections": [[["a0"], ["a1"]]],
                "frames.model2.detections": [[["b0"], ["b1"]]],
                f"frames.{GT}.detections": [[["g0"], ["g1"]]],
                "frames.model1.keyframe": [[True, False]],
            }
        )

        sequence = _fetch(view, ["seq-a"], models=["model1", "model2"])["seq-a"]

        assert sequence.keyframes == {"model1": [True, False]}

    def test_unrelated_query_errors_are_not_retried(self):
        """Only the size limit triggers the split; other failures propagate."""
        view = MagicMock()
        view.match.return_value.values.side_effect = OperationFailure("timeout")

        with pytest.raises(OperationFailure, match="timeout"):
            _fetch(view, ["seq-0", "seq-1"])
        assert view.match.return_value.values.call_count == 1

    def test_process_sequences_batches_the_sequence_list(self):
        names, view = self._view(5)
        processor = _processor(batch_size=2)
        processor.dataset = view
        processor.sequence_list = names

        sequences = processor.process_sequences()

        _, one = self._view(1)
        _fetch(one, ["seq-0"])
        assert sorted(sequences) == sorted(names)
        for name in names:
            assert sequences[name]["model1"] == [["d"]]
            assert sequences[name].keyframes == {"model1": [True]}
        # ceil(5 / batch_size) batches, each as cheap as a single sequence
        assert view.query_count == 3 * one.query_count


@pytest.mark.usefixtures("mock_fiftyone", "mock_compute_payload")
@pytest.mark.parametrize(
    ("overrides", "error", "match"),
    [
        ({"dataset_name": 1}, TypeError, "dataset_name must be of type str"),
        ({"gt_field": None}, TypeError, "gt_field must be of type str"),
        ({"tracking_mode": "yes"}, TypeError, "tracking_mode must be of type bool"),
        ({"tags": "tag1"}, TypeError, "tags must be a list of strings or None"),
        ({"data_type": []}, ValueError, "data_type must be 'rgb' or 'thermal'"),
        (
            {"tracking_mode": True, "models": ["m1", "m2"]},
            ValueError,
            "only one model is supported, but got 2 models",
        ),
        (
            {"tracking_mode": True, "models": ["m1"], "data_type": "rgb"},
            ValueError,
            "not supported for RGB data",
        ),
    ],
)
def test_invalid_parameters_raise(overrides, error, match):
    """Invalid arguments raise the expected error."""
    kwargs = {
        "dataset_name": "valid_dataset",
        "gt_field": "ground_truth_det",
        "models": ["model1"],
        **overrides,
    }
    with pytest.raises(error, match=match):
        PayloadProcessor(**kwargs)


@pytest.fixture
def two_slice_dataset():
    """One sequence filmed by two slices: 2 frames wide, 3 frames narrow.

    Yields:
        fo.Dataset: A throwaway group dataset, deleted after the test.
    """
    dataset = fo.Dataset()
    dataset.add_group_field("group", default="thermal_wide")
    group = fo.Group()
    for name, n_frames in (("thermal_wide", 2), ("thermal_narrow", 3)):
        sample = fo.Sample(
            filepath=f"{name}.mp4",
            group=group.element(name),
            sequence="seq-1",
            metadata=fo.VideoMetadata(frame_width=100, frame_height=50),
        )
        for number in range(1, n_frames + 1):
            sample.frames[number] = fo.Frame(gt=fo.Detections(detections=[]))
        dataset.add_sample(sample)
    yield dataset
    dataset.delete()


def test_a_sequence_with_two_videos_raises(two_slice_dataset):
    with pytest.raises(ValueError, match=r"2 videos .*thermal_narrow', 'thermal_wide"):
        PayloadProcessor(two_slice_dataset.name, "gt", ["gt"])


def test_the_selected_slice_is_recorded_and_fetched(two_slice_dataset):
    processor = PayloadProcessor(
        two_slice_dataset.name, "gt", ["gt"], slices=["thermal_narrow"]
    )
    assert processor.slices == ["thermal_narrow"]
    assert len(processor.payload.sequences["seq-1"]["gt"]) == 3
