# test_payload_processor.py

from unittest.mock import MagicMock, patch

import pytest
from pymongo.errors import OperationFailure

from seametrics.payload import Resolution
from seametrics.payload.processor import PayloadProcessor

# Mock EXCLUDED_CLASSES if necessary
EXCLUDED_CLASSES = ["class1", "class2"]


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

    def __init__(self, values_by_path, too_large_above=None, _stats=None) -> None:
        self._values_by_path = values_by_path
        self._too_large_above = too_large_above
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
        return type(self)(narrowed, self._too_large_above, self._stats)

    def filter_labels(self, *_args: object, **_kwargs: object):
        return self

    @property
    def read_paths(self):
        return self._stats.setdefault("paths", [])

    def values(self, path):
        """One query; a list of paths gives one aligned list per path."""
        self._stats["queries"] += 1
        paths = list(path) if isinstance(path, (list, tuple)) else [path]
        self.read_paths.extend(paths)
        found = [self._value(p) for p in paths]
        return found if isinstance(path, (list, tuple)) else found[0]

    def aggregate(self, aggregations):
        """One query computing several aggregations, as fiftyone batches them."""
        self._stats["queries"] += 1
        return [self._aggregate(agg) for agg in aggregations]

    def _aggregate(self, agg):
        path = agg.field_name
        if type(agg).__name__ != "Count":
            return self._value(path)
        # a count over frames sees the same flags whether stored nested or unwound
        value = self._values_by_path.get(path)
        if value is None:
            value = self._values_by_path.get(
                path.replace("frames.", "frames[].", 1), []
            )
        flat = [x for v in value for x in (v if isinstance(v, list) else [v])]
        return sum(x is not None for x in flat)

    def _value(self, path):
        if path not in self._values_by_path:
            raise ValueError(f"no such field: {path}")
        value = self._values_by_path[path]
        if (
            self._too_large_above is not None
            and isinstance(value, list)
            and len(value) > self._too_large_above
        ):
            raise OperationFailure(
                "Executor error during getMore :: caused by :: BSONObj size:"
                " 17355147 is invalid. Size must be between 0 and 16793600(16MB)"
            )
        return value


class _NestedTooLargeView(_FakeSequenceView):
    """Rejects every nested frame query as oversized; unwound queries fit."""

    def _value(self, path):
        if path.startswith("frames."):
            raise OperationFailure("BSONObj size: 17355147 is invalid.")
        return super()._value(path)


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
        return _FakeSequenceView(flipped, self._too_large_above, self._stats)


class _ReorderedImageView(_ReorderedFieldsView):
    """Image dataset whose label-field queries come back in reverse sample order."""

    media_type = "image"


def _nest(values_by_path: dict, names=("seq-1",)) -> dict:
    """Convert the unwound single-sequence fixtures to the nested batch shape."""
    nested = {
        "id": [f"{name}-id" for name in names],
        "sequence": list(names),
        "metadata.frame_width": [100] * len(names),
        "metadata.frame_height": [50] * len(names),
    }
    for path, value in values_by_path.items():
        nested[path.replace("frames[].", "frames.")] = [value]
    return nested


@pytest.mark.usefixtures("mock_fiftyone", "mock_compute_payload")
class TestProcessSequenceKeyframes:
    GT = "ground_truth_det"

    def _run(self, values_by_path, **kwargs: object):
        processor = _processor(**kwargs)
        processor.dataset = _FakeSequenceView(_nest(values_by_path))
        return processor.process_sequence("seq-1")

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
        assert sequence.keyframes == {"model1": [True, False, True]}
        assert sorted(sequence.field_names) == sorted([self.GT, "model1"])

    def test_tracking_mode_still_blanks_non_keyframes(self):
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


@pytest.mark.usefixtures("mock_fiftyone", "mock_compute_payload")
class TestBatchedFetching:
    """One query per field per batch, with a fallback for oversized results."""

    GT = "ground_truth_det"

    def _view(self, n_sequences, **kwargs: object):
        names = [f"seq-{i}" for i in range(n_sequences)]
        paths = {
            "id": [f"{name}-id" for name in names],
            "sequence": names,
            "metadata.frame_width": [100] * n_sequences,
            "metadata.frame_height": [50] * n_sequences,
            "frames.model1.detections": [[["d"]] for _ in names],
            f"frames.{self.GT}.detections": [[["g"]] for _ in names],
            "frames.model1.keyframe": [[True] for _ in names],
        }
        return names, _FakeSequenceView(paths, **kwargs)

    def test_one_query_per_field_not_per_sequence(self):
        """4 sequences cost the same number of queries as 1."""
        names, view = self._view(4)
        sequences = _fetch(view, names)
        _, one = self._view(1)
        _fetch(one, ["seq-0"])

        assert sorted(sequences) == sorted(names)
        for name in names:
            assert sequences[name]["model1"] == [["d"]]
            assert sequences[name].keyframes == {"model1": [True]}
        assert view.query_count == one.query_count

    def test_frame_range_slices_every_field(self):
        view = _FakeSequenceView(
            _nest(
                {
                    "frames[].model1.detections": [["d0"], ["d1"], ["d2"]],
                    f"frames[].{self.GT}.detections": [["g0"], None, ["g2"]],
                    "frames[].model1.keyframe": [1, None, 1],
                }
            )
        )
        sequence = _fetch(view, ["seq-1"], start_frame_id=1, end_frame_id=3)["seq-1"]
        assert sequence["model1"] == [["d1"], ["d2"]]
        assert sequence[self.GT] == [[], ["g2"]]
        assert sequence.keyframes == {"model1": [False, True]}

    def test_oversized_batch_is_split_and_still_complete(self):
        """A BSON size error halves the batch until the result fits."""
        names, view = self._view(4, too_large_above=2)
        sequences = _fetch(view, names)

        assert sorted(sequences) == sorted(names)

    def test_single_sequence_too_large_still_raises(self):
        """A sequence too large even when unwound raises."""
        names, view = self._view(1, too_large_above=0)
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
                "frames[].model1.detections": [["d0"], ["d1"], ["d2"]],
                f"frames[].{self.GT}.detections": [["g0"], None, ["g2"]],
                "frames[].model1.keyframe": [True, False, True],
            }
        )

        sequence = _fetch(view, ["seq-1"])["seq-1"]

        assert sequence["model1"] == [["d0"], ["d1"], ["d2"]]
        assert sequence[self.GT] == [["g0"], [], ["g2"]]
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
                f"{self.GT}.detections": [["ga0"], ["gb0"], None, ["gb1"]],
                "model1.keyframe": [True, False, True, True],
            }
        )

        sequences = _fetch(view, ["seq-a", "seq-b"])

        assert sequences["seq-a"]["model1"] == [["a0"], ["a1"]]
        assert sequences["seq-a"][self.GT] == [["ga0"], []]
        assert sequences["seq-a"].keyframes == {"model1": [True, True]}
        assert sequences["seq-a"].resolution == Resolution(height=480, width=640)
        assert sequences["seq-b"]["model1"] == [["b0"], []]
        assert sequences["seq-b"][self.GT] == [["gb0"], ["gb1"]]
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
                f"frames.{self.GT}.detections": [[["ga0"]], [["gb0"], ["gb1"]]],
                "frames.model1.keyframe": [[True], [False, True]],
            }
        )

        sequences = _fetch(view, ["seq-a", "seq-b"])

        assert sequences["seq-a"]["model1"] == [["a0"]]
        assert sequences["seq-a"][self.GT] == [["ga0"]]
        assert sequences["seq-b"]["model1"] == [["b0"], ["b1"]]
        assert sequences["seq-b"].keyframes == {"model1": [False, True]}
        assert sequences["seq-b"].resolution == Resolution(height=80, width=200)

    def test_keyframes_read_only_for_models_that_flag_any(self):
        """A model without a single keyframe flag costs no keyframe query."""
        view = _FakeSequenceView(
            {
                "id": ["s1"],
                "sequence": ["seq-a"],
                "metadata.frame_width": [100],
                "metadata.frame_height": [50],
                "frames.model1.detections": [[["a0"], ["a1"]]],
                "frames.model2.detections": [[["b0"], ["b1"]]],
                f"frames.{self.GT}.detections": [[["g0"], ["g1"]]],
                "frames.model1.keyframe": [[True, False]],
                "frames.model2.keyframe": [[None, None]],
            }
        )

        sequence = _fetch(view, ["seq-a"], models=["model1", "model2"])["seq-a"]

        assert sequence.keyframes == {"model1": [True, False]}
        assert "frames.model1.keyframe" in view.read_paths
        assert "frames.model2.keyframe" not in view.read_paths

    def test_unrelated_query_errors_are_not_retried(self):
        """Only the size limit triggers the split; other failures propagate."""
        view = _FakeSequenceView({"sequence": ["seq-0", "seq-1"]})

        with pytest.raises(ValueError, match="no such field"):
            _fetch(view, ["seq-0", "seq-1"])
        assert view.query_count == 1

    def test_process_sequences_batches_the_sequence_list(self):
        names, view = self._view(5)
        processor = _processor(batch_size=2)
        processor.dataset = view
        processor.sequence_list = names

        sequences = processor.process_sequences()

        _, one = self._view(1)
        _fetch(one, ["seq-0"])
        assert sorted(sequences) == sorted(names)
        assert view.query_count == 3 * one.query_count  # ceil(5 / batch_size)


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
