import logging
from typing import List, Dict, Literal
from tqdm import tqdm
import fiftyone as fo
from fiftyone import ViewField as F

from seametrics.payload import Payload, Resolution, Sequence
from seametrics.constants import EXCLUDED_CLASSES

# Set up logging
logger = logging.getLogger(__name__)


class PayloadProcessor:
    """
    Class to process a payload and generate sequence data.
    """

    def __init__(
        self,
        dataset_name: str,
        gt_field: str,
        models: List[str],
        tracking_mode: bool = False,
        sequence_list: List[str] = None,
        data_type: Literal["rgb", "thermal"] = "thermal",
        excluded_classes: List[str] = None,
    ):
        """
        Initializes a PayloadProcessor object.

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
            excluded_classes (List[str], optional): The list of excluded classes.
                Defaults to None.
        """
        self.dataset_name = dataset_name
        self.gt_field = gt_field
        self.models = models
        self.tracking_mode = tracking_mode
        self.sequence_list = sequence_list
        self.data_type = data_type
        self.excluded_classes = excluded_classes or EXCLUDED_CLASSES
        self.validate_input_parameters(dataset_name)
        self.dataset: fo.Dataset = None
        self.payload: Payload = None
        self.compute_payload()
        logger.info(f"Initialized PayloadProcessor for dataset: {dataset_name}")

    def compute_payload(self) -> Payload:
        """
        Recomputes the payload after updating any of the input parameters.

        Returns:
            Payload: The updated payload.
        """
        self.validate_input_parameters(self.dataset_name)
        self.dataset = fo.load_dataset(self.dataset_name)
        if not self.sequence_list:
            self.sequence_list = self.dataset.distinct("sequence")
        logger.debug(f"{self.dataset}")

        self.payload = Payload(
            dataset=self.dataset.name,
            models=self.models,
            gt_field_name=self.gt_field,
            sequences=self.process_dataset(),
        )
        return self.payload

    def validate_input_parameters(self, dataset_name: str):
        """
        Validates the input parameters.

        Args:
            dataset_name (str): The name of the dataset.

        Raises:
            ValueError: If dataset is not found or tracking mode enabled for RGB data.
        """
        if dataset_name not in fo.list_datasets():
            raise ValueError(f"Dataset {dataset_name} not found in FiftyOne.")
        if self.tracking_mode and self.data_type == "rgb":
            raise ValueError("Tracking-mode evaluation is not supported for RGB data.")

    def process_dataset(self):
        """
        Processes the dataset and generates sequence data.

        Returns:
            Dict: A dictionary containing sequence data.
        """
        self.print_info()

        if len(self.sequence_list) == 0:
            self.sequence_list = self.dataset.distinct("sequence")
            logger.info(f"Using all sequences in dataset: {self.sequence_list}")

        relevant_slices = self.get_datatype_slices()
        self.dataset = self.dataset.select_group_slices(relevant_slices)
        logger.info(f"Using slice: {relevant_slices}")

        return self.process_sequences()

    def get_datatype_slices(self) -> List[str]:
        """
        Retrieves the relevant slices based on the data type.

        Returns:
            List[str]: The list of relevant slices.

        Raises:
            ValueError: If there is no matching data slice for the data type.
        """
        thermal_slices = {"thermal_wide", "thermal_right", "thermal_left"}
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
        """
        Retrieves the resolution from a FiftyOne view.

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
    def get_field_name(view: fo.DatasetView, field_name: str, unwinding: bool = False) -> str:
        """
        Retrieves the field name based on the media type of the view.

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
        num_samples: int = None,
    ) -> Sequence:
        """
        Retrieves the sequence data from the dataset view.

        Args:
            sequence (str): The name of the sequence.

        Returns:
            Sequence: The sequence data.
        """
        sequence_view = self.dataset.match(F("sequence") == sequence).filter_labels(
            self.get_field_name(self.dataset, self.gt_field),
            ~F("label").is_in(self.excluded_classes),
            only_matches=False,
        ).take(num_samples)

        detections = {}
        for field_name in self.models + [self.gt_field]:
            field = self.get_field_name(sequence_view, field_name, unwinding=True)
            det_values = sequence_view.filter_labels(
                        field,
                        ~F("label").is_in(self.excluded_classes),
                        only_matches=False,
                    ).values(f"{field}.detections")
            detections[field_name] = [
                d if d is not None else []
                for d in det_values
            ]
        return Sequence(resolution=self.get_resolution(sequence_view), **detections)

    def process_sequences(self) -> Dict[str, Sequence]:
        """
        Processes the sequences and generates sequence data.

        Returns:
            Dict: A dictionary containing sequence data.
        """
        sequences = {}
        for sequence in tqdm(self.sequence_list, desc="Processing sequences"):
            seq_data = self.process_sequence(sequence)
            if seq_data:
                sequences[sequence] = seq_data
        return sequences

    def print_info(self):
        """
        Prints information about the payload processor.
        """
        logger.info(f"Processing dataset: {self.dataset.name}")
        logger.info(f"GT field: {self.gt_field}")
        logger.info(f"Models: {self.models}")
        logger.info(f"Tracking mode: {self.tracking_mode}")
        logger.info(f"Sequence list: {self.sequence_list}")
        logger.info(f"Data type: {self.data_type}")
        logger.info(f"Excluded classes: {self.excluded_classes}")


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
