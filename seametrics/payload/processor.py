"""Build evaluation payloads from FiftyOne datasets."""

import logging
from typing import Dict, List, Literal, Optional

import fiftyone as fo
from fiftyone import ViewField as F
from tqdm import tqdm

from seametrics.constants import EXCLUDED_CLASSES
from seametrics.payload import Payload, Resolution, Sequence

# Set up logging
logger = logging.getLogger(__name__)


def _is_str_list(value: object) -> bool:
    """Return True if *value* is a list of strings."""
    return isinstance(value, list) and all(isinstance(v, str) for v in value)


# Public constructor: every argument is part of the API and stored on the instance.
class PayloadProcessor:  # pylint: disable=too-many-instance-attributes
    """Builds a Payload of per-sequence detections from a FiftyOne dataset."""

    def __init__(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        dataset_name: str,
        gt_field: str,
        models: List[str],
        tracking_mode: bool = False,
        sequence_list: Optional[List[str]] = None,
        data_type: Literal["rgb", "thermal"] = "thermal",
        excluded_classes: Optional[List[str]] = None,
        slices: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        start_frame_id: Optional[int] = None,
        end_frame_id: Optional[int] = None,
        confidence_threshold: float = 0,
    ) -> None:
        """Initializes a PayloadProcessor object.

        Args:
            dataset_name (str): The name of the dataset.
            gt_field (str): The name of the ground truth field.
            models (List[str]): The list of model names.
            tracking_mode (bool, optional): Whether to enable tracking mode.
                Defaults to False.
            sequence_list (List[str], optional): The list of sequence names.
                Defaults to None.
            data_type (Literal["rgb", "thermal"], optional): The type of data.
                Defaults to "thermal".
            excluded_classes (List[str], optional): Classes dropped from every
                field. Defaults to ``EXCLUDED_CLASSES``.
            slices (List[str], optional): Group slices to process. If None, uses
                the dataset's slices matching ``data_type``.
            tags (List[str], optional): The list of tags to filter the dataset.
                Defaults to None.
            start_frame_id (int, optional): 0-based index of the first frame to
                keep. Defaults to None.
            end_frame_id (int, optional): 0-based index of the last frame to keep
                (inclusive). Defaults to None.
            confidence_threshold (float, optional): Confidence threshold to filter
                the model fields. Defaults to 0.
        """
        self.dataset_name = dataset_name
        self.gt_field = gt_field
        self.models = models
        self.tracking_mode = tracking_mode
        self.sequence_list = sequence_list
        self.data_type = data_type
        self.slices = slices
        self.tags = tags
        self.confidence_threshold = confidence_threshold
        self.excluded_classes = excluded_classes or EXCLUDED_CLASSES
        self.validate_input_parameters()
        self.dataset: fo.Dataset = None
        self.payload: Payload = None
        self.start_frame_id = start_frame_id
        self.end_frame_id = end_frame_id + 1 if end_frame_id is not None else None
        self.compute_payload()
        logger.info(f"Initialized PayloadProcessor for dataset: {dataset_name}")

    def compute_payload(self) -> Payload:
        """Recomputes the payload after updating any of the input parameters.

        Returns:
            Payload: The updated payload.
        """
        self.validate_input_parameters()
        self.dataset = fo.load_dataset(self.dataset_name)
        logger.debug(f"{self.dataset}")

        self.payload = Payload(
            dataset=self.dataset.name,
            models=self.models,
            gt_field_name=self.gt_field,
            sequences=self.process_dataset(),
        )
        return self.payload

    def validate_input_parameters(self) -> None:
        """Validates the input parameters.

        ``_validate_types`` raises TypeError for a parameter of the wrong type.

        Raises:
            ValueError: If ``data_type`` is not "rgb" or "thermal", the dataset is
                not found in FiftyOne, or tracking mode is enabled for RGB data or
                with more than one model.
        """
        self._validate_types()

        if self.tracking_mode and len(self.models) != 1:
            raise ValueError(
                "When tracking mode is enabled only one model is supported,"
                f" but got {len(self.models)} models"
            )

        # Check data_type is either "rgb" or "thermal"
        if not isinstance(self.data_type, str) or self.data_type not in {
            "rgb",
            "thermal",
        }:
            raise ValueError(
                f"data_type must be 'rgb' or 'thermal', but got {self.data_type}"
            )

        # Additional validation logic can go here, like dataset name lookup
        if self.dataset_name not in fo.list_datasets():
            raise ValueError(f"Dataset {self.dataset_name} not found in FiftyOne.")

        if self.tracking_mode and self.data_type == "rgb":
            raise ValueError("Tracking-mode evaluation is not supported for RGB data.")

    def _validate_types(self) -> None:
        """Checks the type of every input parameter.

        Raises:
            TypeError: If any of the parameters are of the wrong type.
        """
        if not isinstance(self.dataset_name, str):
            raise TypeError(
                f"dataset_name must be of type str, but got {type(self.dataset_name)}"
            )

        if not isinstance(self.gt_field, str):
            raise TypeError(
                f"gt_field must be of type str, but got {type(self.gt_field)}"
            )

        if not _is_str_list(self.models):
            raise TypeError(f"models must be a list of strings, but got {self.models}")

        if not isinstance(self.tracking_mode, bool):
            raise TypeError(
                "tracking_mode must be of type bool,"
                f" but got {type(self.tracking_mode)}"
            )

        for name in ("sequence_list", "excluded_classes", "slices", "tags"):
            value = getattr(self, name)
            if value is not None and not _is_str_list(value):
                raise TypeError(
                    f"{name} must be a list of strings or None, but got {value}"
                )

    def process_dataset(self) -> Dict[str, Sequence]:
        """Processes the dataset and generates sequence data.

        Returns:
            Dict: A dictionary containing sequence data.
        """
        self.print_info()

        if self.dataset.media_type == "group":
            if not self.slices:
                relevant_slices = self.get_datatype_slices()
            else:
                relevant_slices = set(self.slices)

            logger.info(f"Using slice: {relevant_slices}")

            self.dataset = self.dataset.select_group_slices(relevant_slices)

        if self.tags:
            self.dataset = self.dataset.match_tags(self.tags, all=True)

        if self.sequence_list:
            self.dataset = self.dataset.match(F("sequence").is_in(self.sequence_list))

        self.sequence_list = self.dataset.distinct("sequence")

        return self.process_sequences()

    def get_datatype_slices(self) -> List[str]:
        """Retrieves the relevant slices based on the data type.

        Returns:
            List[str]: The list of relevant slices.

        Raises:
            ValueError: If there is no matching data slice for the data type.
        """
        thermal_slices = {
            "thermal_wide",
            "thermal_narrow",
            "thermal_right",
            "thermal_left",
            "thermal_stitched",
        }
        rgb_slices = {"rgb", "rgb_wide", "rgb_narrow"}

        existing_slices = set(self.dataset.group_slices)
        if self.data_type == "thermal":
            chosen_slices = list(thermal_slices & existing_slices)
        elif self.data_type == "rgb":
            chosen_slices = list(rgb_slices & existing_slices)
            if not chosen_slices:
                raise ValueError("RGB data cannot be evaluated in video data.")
        else:
            raise ValueError(f"No matching data slice for datatype: {self.data_type}.")

        if not chosen_slices:
            raise ValueError(f"No slice corresponding to datatype: {self.data_type}.")

        return chosen_slices

    @staticmethod
    def get_resolution(view: fo.DatasetView) -> Resolution:
        """Retrieves the resolution from a FiftyOne view.

        Args:
            view (fo.DatasetView): The FiftyOne dataset view.

        Returns:
            Resolution: The resolution of the sequence.
        """
        if view.media_type == "video":
            height = view.first().metadata.frame_height
            width = view.first().metadata.frame_width
        elif view.media_type == "image":
            height = view.first().metadata.height
            width = view.first().metadata.width
        else:
            raise ValueError(f"Unsupported media type: {view.media_type}")
        return Resolution(height=height, width=width)

    @staticmethod
    def get_field_name(
        view: fo.DatasetView, field_name: str, unwinding: bool = False
    ) -> str:
        """Retrieves the field name based on the media type of the view.

        Args:
            view (fo.DatasetView): The FiftyOne dataset view.
            field_name (str): The field name.
            unwinding (bool, optional): Whether to unwind the field. Defaults to False.

        Returns:
            str: The field name.
        """
        if view.media_type == "video":
            return f"frames[].{field_name}" if unwinding else f"frames.{field_name}"
        if view.media_type == "image":
            return field_name
        raise ValueError(f"Unsupported media type: {view.media_type}")

    def process_sequence(
        self,
        sequence: str,
    ) -> Sequence:
        """Retrieves the sequence data from the dataset view.

        Args:
            sequence (str): The name of the sequence.

        Returns:
            Sequence: The sequence data.
        """
        sequence_view = self.dataset.match(F("sequence") == sequence).filter_labels(
            self.get_field_name(self.dataset, self.gt_field),
            ~F("label").is_in(self.excluded_classes),
            only_matches=False,
        )

        detections = {}

        for field_name in [*self.models, self.gt_field]:
            filter_expression = ~(F("label").is_in(self.excluded_classes))

            if self.confidence_threshold > 0 and field_name != self.gt_field:
                filter_expression &= F("confidence") > self.confidence_threshold

            filter_view = sequence_view.filter_labels(
                self.get_field_name(sequence_view, field_name),
                filter_expression,
                only_matches=False,
            )

            det_field = self.get_field_name(sequence_view, field_name, unwinding=True)
            det_values = filter_view.values(f"{det_field}.detections")[
                self.start_frame_id : self.end_frame_id
            ]

            if self.tracking_mode:
                kf_field = self.get_field_name(
                    sequence_view, self.models[0], unwinding=True
                )
                keyframe_values = filter_view.values(f"{kf_field}.keyframe")[
                    self.start_frame_id : self.end_frame_id
                ]
                detections[field_name] = [
                    d if d is not None and k else []
                    for d, k in zip(det_values, keyframe_values, strict=False)
                ]
            else:
                detections[field_name] = [
                    d if d is not None else [] for d in det_values
                ]

        return Sequence(resolution=self.get_resolution(sequence_view), **detections)

    def process_sequences(self) -> Dict[str, Sequence]:
        """Processes the sequences and generates sequence data.

        Returns:
            Dict: A dictionary containing sequence data.
        """
        sequences = {}
        for sequence in tqdm(self.sequence_list, desc="Processing sequences"):
            seq_data = self.process_sequence(sequence)
            if seq_data:
                sequences[sequence] = seq_data
        return sequences

    def print_info(self) -> None:
        """Logs the processor's settings at INFO level."""
        logger.info(f"Processing dataset: {self.dataset.name}")
        logger.info(f"GT field: {self.gt_field}")
        logger.info(f"Models: {self.models}")
        logger.info(f"Tracking mode: {self.tracking_mode}")
        logger.info(f"Sequence list: {self.sequence_list}")
        logger.info(f"Data type: {self.data_type}")
        logger.info(f"Excluded classes: {self.excluded_classes}")
        logger.info(f"Slices: {self.slices}")


# Setup logging at the beginning of your script
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    processor = PayloadProcessor(
        dataset_name="SAILING_DATASET_QA",
        gt_field="ground_truth_det",
        models=["yolov5n6_RGB_D2304-v1_9C"],
        tracking_mode=False,
        sequence_list=["Trip_14_Seq_1", "Trip_14_Seq_2"],
        data_type="thermal",
    )
    processed_payload = processor.process_dataset()
