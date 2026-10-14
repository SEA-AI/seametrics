import logging
from typing import Dict, List, Literal, Union

import fiftyone as fo
from fiftyone import ViewField as F
from tqdm import tqdm

from seametrics.constants import EXCLUDED_CLASSES
from seametrics.payload import Payload, Resolution, Sequence

# Set up logging
logger = logging.getLogger(__name__)


def _is_result_too_large(exc: Exception) -> bool:
    """Return True if *exc* is MongoDB rejecting a result for exceeding 16 MB."""
    message = str(exc)
    return "BSONObj size" in message or "BSONObjectTooLarge" in message


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
        slices: List[str] = None,
        tags: List[str] = None,
        start_frame_id: int = None,
        end_frame_id: int = None,
        confidence_threshold: float = 0,
        batch_size: int = 8,
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
            slices (List[str], optional): The list of slices to process.
                Defaults to None. If None, a smart selection of available slices takes place.
            tags (List[str], optional): The list of tags to filter the dataset.
                Defaults to None.
            start_frame_id (int, optional): The start frame id.
                Defaults to None.
            end_frame_id (int, optional): The end frame id.
                Defaults to None.
            confidence_threshold (float, optional): Confidence threshold to filter the model fields.
                Defaults to 0.
            batch_size (int, optional): How many sequences to fetch per query. Larger
                batches mean fewer queries but more memory, and are split
                automatically when they exceed MongoDB's 16 MB result limit.
                Defaults to 8.
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
        self.batch_size = batch_size
        self.excluded_classes = excluded_classes or EXCLUDED_CLASSES
        self.validate_input_parameters()
        self.dataset: fo.Dataset = None
        self.payload: Payload = None
        self.start_frame_id = start_frame_id
        self.end_frame_id = end_frame_id + 1 if end_frame_id is not None else None
        self.compute_payload()
        logger.info(f"Initialized PayloadProcessor for dataset: {dataset_name}")

    def compute_payload(self) -> Payload:
        """
        Recomputes the payload after updating any of the input parameters.

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

    def validate_input_parameters(self):
        """
        Validates the input parameters.

        Raises:
            TypeError: If any of the parameters are of the wrong type.
            ValueError: If dataset is not found in FiftyOne or if tracking mode is enabled for RGB data.

        Validations:
            - `dataset_name` must be a string.
            - `gt_field` must be a string.
            - `models` must be a list of strings.
            - `tracking_mode` must be a boolean.
            - `sequence_list` must be None or a list of strings.
            - `data_type` must be either "rgb" or "thermal".
            - `excluded_classes` must be None or a list of strings.
            - `slices` must be None or a list of strings.
            - `tags` must be None or a list of strings
        """
        
        # Check dataset_name is a string
        if not isinstance(self.dataset_name, str):
            raise TypeError(f"dataset_name must be of type str, but got {type(self.dataset_name)}")

        # Check gt_field is a string
        if not isinstance(self.gt_field, str):
            raise TypeError(f"gt_field must be of type str, but got {type(self.gt_field)}")

        # Check models is a list of strings
        if not isinstance(self.models, list) or not all(isinstance(model, str) for model in self.models):
            raise TypeError(f"models must be a list of strings, but got {self.models}")

        # Check tracking_mode is a boolean
        if not isinstance(self.tracking_mode, bool):
            raise TypeError(f"tracking_mode must be of type bool, but got {type(self.tracking_mode)}")
        
        if self.tracking_mode and len(self.models) != 1:
            raise ValueError(f"When tracking mode is enabled only one model is supported, but got {len(self.models)} models")

        # Check sequence_list is None or a list of strings
        if self.sequence_list is not None and (not isinstance(self.sequence_list, list) or not all(isinstance(seq, str) for seq in self.sequence_list)):
            raise TypeError(f"sequence_list must be a list of strings or None, but got {self.sequence_list}")

        # Check data_type is either "rgb" or "thermal"
        if self.data_type not in ["rgb", "thermal"]:
            raise ValueError(f"data_type must be 'rgb' or 'thermal', but got {self.data_type}")

        # Check excluded_classes is None or a list of strings
        if self.excluded_classes is not None and (not isinstance(self.excluded_classes, list) or not all(isinstance(cls, str) for cls in self.excluded_classes)):
            raise TypeError(f"excluded_classes must be a list of strings or None, but got {self.excluded_classes}")

        # Check slices is None or a list of strings
        if self.slices is not None and (not isinstance(self.slices, list) or not all(isinstance(s, str) for s in self.slices)):
            raise TypeError(f"slices must be a list of strings or None, but got {self.slices}")

        # Check tags is None or a list of strings
        if self.tags is not None and (not isinstance(self.tags, list) or not all(isinstance(tag, str) for tag in self.tags)):
            raise TypeError(f"tags must be a list of strings or None, but got {self.tags}")

        # Additional validation logic can go here, like dataset name lookup
        if self.dataset_name not in fo.list_datasets():
            raise ValueError(f"Dataset {self.dataset_name} not found in FiftyOne.")

        if self.tracking_mode and self.data_type == "rgb":
            raise ValueError("Tracking-mode evaluation is not supported for RGB data.")

    def process_dataset(self):
        """
        Processes the dataset and generates sequence data.

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
        """
        Retrieves the relevant slices based on the data type.

        Returns:
            List[str]: The list of relevant slices.

        Raises:
            ValueError: If there is no matching data slice for the data type.
        """        
        thermal_slices = {"thermal_wide", "thermal_narrow", "thermal_right", "thermal_left", "thermal_stitched"}
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
    def get_field_name(
        view: fo.DatasetView, field_name: str, unwinding: bool = False
    ) -> str:
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

    def process_sequence(self, sequence: str) -> Sequence:
        """Retrieve the sequence data from the dataset view.

        Args:
            sequence (str): The name of the sequence.

        Returns:
            Sequence: The sequence data.
        """
        return self.fetch_sequences([sequence])[sequence]

    def label_filter(self, field_name: str) -> object:
        """Build the label filter expression for one field.

        Args:
            field_name (str): Detection field the filter applies to.

        Returns:
            ViewExpression: Excludes the configured classes, and for prediction
                fields also drops detections below the confidence threshold.
        """
        expression = ~(F("label").is_in(self.excluded_classes))
        if self.confidence_threshold > 0 and field_name != self.gt_field:
            expression &= F("confidence") > self.confidence_threshold
        return expression

    @staticmethod
    def normalize_keyframes(values: list) -> "List[bool] | None":
        """Turn raw keyframe values for one sequence into a mask.

        The flags are recorded on the `Sequence` rather than applied, so each
        metric family can decide what to do with them: `seametrics.tracking`
        evaluates keyframes only, while detection evaluates every frame unless the
        caller opts in.

        Args:
            values (list): Raw per-frame keyframe values, already sliced to the
                configured frame range.

        Returns:
            List[bool] | None: One flag per frame, or None when the field carries
                no usable keyframe data (no attribute, or no frame flagged) — the
                common case for a plain detection model.
        """
        if not values or not any(values):
            return None
        return [bool(value) for value in values]

    def group_by_sequence(self, names: list, values: list) -> Dict[str, list]:
        """Group per-sample values by sequence name.

        For video data every sample already holds a whole sequence, so its value
        is a list of frames. For image data each sample is a single frame and the
        samples of a sequence have to be collected in order.

        Args:
            names (list): Sequence name of each sample, in view order.
            values (list): Value of each sample, in the same order.

        Returns:
            Dict[str, list]: Per-frame values keyed by sequence name.
        """
        if values is None:
            return {}
        if self.dataset.media_type == "video":
            return {name: (values[i] or []) for i, name in enumerate(names)}
        grouped: Dict[str, list] = {}
        for name, value in zip(names, values, strict=True):
            grouped.setdefault(name, []).append(value)
        return grouped

    def group_values(self, names: list, values: list, unwound: bool) -> Dict[str, list]:
        """Group fetched values by sequence name.

        Args:
            names (list): Sequence name of each sample, in view order.
            values (list): Fetched values.
            unwound (bool): True when the query unwound frames, in which case
                *values* is already the flat frame list of the single sequence.

        Returns:
            Dict[str, list]: Per-frame values keyed by sequence name.
        """
        if unwound:
            return {names[0]: values or []}
        return self.group_by_sequence(names, values)

    def fetch_sequences(self, batch: List[str]) -> Dict[str, Sequence]:
        """Fetch a batch of sequences with one query per field.

        Querying once per field for a whole batch is far cheaper than querying
        per sequence, but the aggregation result has to fit inside MongoDB's 16 MB
        BSON limit. Dense sequences exceed it, so rather than guessing a safe batch
        size the batch is halved whenever the server rejects the result, down to a
        single sequence.

        Args:
            batch (List[str]): Sequence names to fetch.

        Returns:
            Dict[str, Sequence]: The fetched sequences, keyed by name.

        Note:
            Any query error that is not the BSON size limit propagates unchanged,
            as does the size limit itself once a single sequence is still too big.
        """
        try:
            return self._fetch_sequences(batch)
        except Exception as exc:  # pylint: disable=broad-except
            if not _is_result_too_large(exc):
                raise
            if len(batch) == 1:
                # A nested query packs a whole sequence into one BSON document,
                # which cannot exceed 16 MB. Unwinding streams the frames as
                # separate documents instead, so a dense sequence still fits.
                return self._fetch_sequences(batch, unwound=True)
            middle = len(batch) // 2
            logger.debug(
                f"Batch of {len(batch)} exceeded the BSON limit; splitting in two"
            )
            fetched = self.fetch_sequences(batch[:middle])
            fetched.update(self.fetch_sequences(batch[middle:]))
            return fetched

    def _fetch_sequences(
        self, batch: List[str], unwound: bool = False
    ) -> Dict[str, Sequence]:
        """Fetch a batch of sequences without the size fallback.

        Args:
            batch (List[str]): Sequence names to fetch.
            unwound (bool): Query frames as separate documents rather than one
                document per sequence. Sidesteps the per-document BSON limit, and
                is only valid for a single sequence.

        Returns:
            Dict[str, Sequence]: The fetched sequences, keyed by name.
        """
        view = self.dataset.match(F("sequence").is_in(batch))
        names = view.values("sequence")
        if not names:
            return {}

        frame_range = slice(self.start_frame_id, self.end_frame_id)
        detections: Dict[str, Dict[str, list]] = {}
        keyframes: Dict[str, Dict[str, list]] = {}

        for field_name in [*self.models, self.gt_field]:
            filter_view = view.filter_labels(
                self.get_field_name(view, field_name),
                self.label_filter(field_name),
                only_matches=False,
            )
            path = self.get_field_name(view, field_name, unwinding=unwound)
            detections[field_name] = self.group_values(
                names, filter_view.values(f"{path}.detections"), unwound
            )
            if field_name == self.gt_field:
                continue
            try:
                raw = filter_view.values(f"{path}.keyframe")
            except Exception:  # pylint: disable=broad-except
                continue  # field has no `keyframe` attribute; nothing to record
            keyframes[field_name] = self.group_values(names, raw, unwound)

        resolutions = self.get_resolutions(view, names)
        keyframe_source = self.models[0] if self.tracking_mode else None

        sequences = {}
        for name in dict.fromkeys(names):
            fields = {
                field: [frame or [] for frame in per_seq.get(name, [])][frame_range]
                for field, per_seq in detections.items()
            }
            masks = {}
            for field, per_seq in keyframes.items():
                mask = self.normalize_keyframes(per_seq.get(name, [])[frame_range])
                if mask is not None:
                    masks[field] = mask

            if keyframe_source is not None:
                blanking = masks.get(keyframe_source)
                if blanking is not None:
                    fields = {
                        field: [
                            frame if keep else []
                            for frame, keep in zip(frames, blanking, strict=False)
                        ]
                        for field, frames in fields.items()
                    }

            sequences[name] = Sequence(
                resolution=resolutions[name], keyframes=masks, **fields
            )
        return sequences

    @staticmethod
    def get_resolutions(view: fo.DatasetView, names: list) -> Dict[str, Resolution]:
        """Read the frame resolution of every sequence in a view.

        Args:
            view (fo.DatasetView): View covering the sequences.
            names (list): Sequence name of each sample, in view order.

        Returns:
            Dict[str, Resolution]: Resolution keyed by sequence name.

        Raises:
            ValueError: If the media type of *view* is not "video" or "image".
        """
        if view.media_type == "video":
            widths = view.values("metadata.frame_width")
            heights = view.values("metadata.frame_height")
        elif view.media_type == "image":
            widths = view.values("metadata.width")
            heights = view.values("metadata.height")
        else:
            raise ValueError(f"Unsupported media type: {view.media_type}")
        return {
            name: Resolution(height=heights[i], width=widths[i])
            for i, name in enumerate(names)
        }

    def process_sequences(self) -> Dict[str, Sequence]:
        """Process the sequences and generate sequence data.

        Sequences are fetched in batches so that each field costs one query per
        batch rather than one per sequence.

        Returns:
            Dict: A dictionary containing sequence data.
        """
        sequences = {}
        batches = [
            self.sequence_list[i : i + self.batch_size]
            for i in range(0, len(self.sequence_list), self.batch_size)
        ]
        for batch in tqdm(batches, desc="Processing sequences"):
            sequences.update(self.fetch_sequences(batch))
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
