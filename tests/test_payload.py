import pytest
from seametrics.payload import Sequence, Resolution
from seametrics.payload.processor import PayloadProcessor


@pytest.fixture
def sample_resolution():
    return Resolution(height=1080, width=1920)


@pytest.fixture
def sample_sequence(sample_resolution: Resolution):
    return Sequence(resolution=sample_resolution, attr1="value1", attr2="value2")


def test_sequence_attributes(sample_sequence: Sequence):
    assert sample_sequence.resolution.height == 1080
    assert sample_sequence.resolution.width == 1920
    assert sample_sequence.attr1 == "value1"
    assert sample_sequence.attr2 == "value2"


# def test_payload_processor_init():
#     processor = PayloadProcessor(
#         dataset_name="test_dataset",
#         gt_field="ground_truth",
#         models=["model1", "model2"],
#         tracking_mode=False,
#         sequence_list=["sequence1", "sequence2"],
#         data_type="thermal",
#         excluded_classes=["class1", "class2"],
#     )

#     assert processor.gt_field == "ground_truth"
#     assert processor.models == ["model1", "model2"]
#     assert processor.tracking_mode is False
#     assert processor.sequence_list == ["sequence1", "sequence2"]
#     assert processor.data_type == "thermal"
#     assert processor.excluded_classes == ["class1", "class2"]
#     assert processor.dataset == "test_dataset"


def test_payload_processor_get_resolution():
    # Create a mock dataset view
    class MockDatasetView:
        def __init__(self, media_type, height, width):
            self.media_type = media_type
            self.height = height
            self.width = width
            self.frame_height = height
            self.frame_width = width

        def first(self):
            """use for first().metadata.frame_height and first().metadata.frame_width"""
            return self

        @property
        def metadata(self):
            return self

    view = MockDatasetView(media_type="video", height=720, width=1280)
    resolution = PayloadProcessor.get_resolution(view)

    assert resolution.height == 720
    assert resolution.width == 1280


def test_payload_processor_get_field_name():
    # Create a mock dataset view
    class MockDatasetView:
        def __init__(self, media_type):
            self.media_type = media_type

    view = MockDatasetView(media_type="video")

    field_name = PayloadProcessor.get_field_name(view, "detections")
    assert field_name == "frames.detections"

    field_name = PayloadProcessor.get_field_name(view, "detections", unwinding=True)
    assert field_name == "frames[].detections"

    view = MockDatasetView(media_type="image")
    
    field_name = PayloadProcessor.get_field_name(view, "detections")
    assert field_name == "detections"

    field_name = PayloadProcessor.get_field_name(view, "detections", unwinding=True)
    assert field_name == "detections"
