import motmetrics as mm
import numpy as np
from motmetrics.metrics import events_to_df_map, obj_frequencies, track_ratios
from seametrics.payload import Payload, Sequence
from typing import Dict, List, Tuple

def validate_inputs(predictions, references) -> Tuple:
    """
    Validate the inputs to the calculate function.

    Parameters:
        predictions (list): A list of lists in the format [frame number, object id, bb_left, bb_top, bb_width, bb_height, confidence].
        references (list): A list of lists in the format [frame number, object id, bb_left, bb_top, bb_width, bb_height].

    Returns:
        tuple: A tuple containing the validated predictions and references.
    """

    try:
        np_predictions = np.array(predictions) if predictions else np.empty((0, 7))
    except:
        raise ValueError(
            "The predictions should be a list of np.arrays in the format [frame number, object id, bb_left, bb_top, bb_width, bb_height, confidence]"
        )

    try:
        np_references = np.array(references) if references else np.empty((0, 6))
    except:
        raise ValueError(
            "The references should be a list of np.arrays in the format [frame number, object id, bb_left, bb_top, bb_width, bb_height]"
        )

    if np_predictions.ndim < 2 or np_predictions.shape[1] != 7:
        raise ValueError(
            "The predictions should be a 2D array with 7 columns in the format [frame number, object id, bb_left, bb_top, bb_width, bb_height, confidence]"
        )

    if np_references.ndim < 2 or np_references.shape[1] != 6:
        raise ValueError(
            "The references should be a 2D array with 6 columns in the format [frame number, object id, bb_left, bb_top, bb_width, bb_height]"
        )

    if np_predictions.size > 0:
        if np_predictions[:, 0].min() <= 0:
            raise ValueError(
                "The frame number in the predictions should be a positive integer"
            )
    if np_references.size > 0:
        if np_references[:, 0].min() <= 0:
            raise ValueError(
                "The frame number in the references should be a positive integer"
            )

    return np_predictions, np_references

def get_formated_references(frames, filter) -> Dict:
    """
    Formats the references for the calculate_from_payload function, based on the filter and its ranges

    Parameters:
        frames (list): A list of lists in the format [frame number, object id, bb_left, bb_top, bb_width, bb_height].
        filter (dict): A dictionary containing the filter name and its ranges.

    Returns:
        dict: A dictionary containing the formated references in list of lists format.
    """

    filter_name = filter["name"]
    filter_ranges = filter["ranges"]
    formated_references = {}

    for filter_range in filter_ranges:
        filter_range_name = filter_range[0]
        formated_references[filter_range_name] = []

    for frame_id, frame in enumerate(frames):
        for detection in frame:
            index = detection["index"]
            x, y, w, h = detection["bounding_box"]
            filter_value = detection[filter_name]

            for filter_range in filter_ranges:
                filter_range_name, filter_range_limits = (
                    filter_range[0],
                    filter_range[1],
                )
                if (
                    filter_value >= filter_range_limits[0]
                    and filter_value <= filter_range_limits[1]
                ):
                    formated_references[filter_range_name].append(
                        [frame_id + 1, index, x, y, w, h]
                    )

    return formated_references

def get_formated_predictions(frames) -> List[List]:
    """
    Formats the predictions for the calculate_from_payload function
    
    Parameters:
        frames (list): A list of lists of fo.Detection

    Returns:
        list: A list of lists in the format [frame number, object id, bb_left, bb_top, bb_width, bb_height, confidence].

    """
    formated_predictions = []
    for frame_id, frame in enumerate(frames):
        for detection in frame:
            index = detection["index"]
            x, y, w, h = detection["bounding_box"]
            confidence = 1
            formated_predictions.append([frame_id + 1, index, x, y, w, h, confidence])

    return formated_predictions

def payload_to_uf_metrics(
    payload: Payload,
    model_name: str = None,
    filter_dict = {"name": "area", "ranges": [("all", [0, 1e5**2])]},
    ) -> Tuple[List[np.ndarray], List[Dict[str, np.ndarray]]]:
    """
    Convert the payload data to UserFriendly metrics format.

    Parameters:
        payload (dict): The payload data containing sequences, models,

    Returns:
        tuple: A tuple containing the converted (predictions, references).

    """

    predictions, references = [], []

    for sequence_name, sequence in payload.sequences.items():
        formated_predictions = get_formated_predictions(sequence[model_name])
        formated_references = get_formated_references(sequence[payload.gt_field_name], filter_dict)
        predictions.append(formated_predictions)
        references.append(formated_references)

    return predictions, references
    

class UFM:
    """
    Class for computing UserFriendly metrics.

    Methods
    -------
    compute()
        Compute the UserFriendly metrics.

    """

    def __init__(
        self, 
        iou_threshold: float = 1e-10, 
        recognition_thresholds: list = [0.3, 0.5, 0.8],
    ):
        """
        Initialize the UserFriendly class.

        Parameters
        ----------
        payload : Payload
            The payload object.
        max_iou : float
            The maximum intersection over union (IoU) threshold.
        filters : dict
            A dictionary of filters to apply to the data.
        recognition_thresholds : list
            A list of recognition thresholds to use.
        debug : bool
            Whether to print debug messages.

        """
        self.iou_threshold = iou_threshold
        self.recognition_thresholds = recognition_thresholds

    def motmetrics_compute(
        self, 
        np_predictions, 
        np_references, 
        iou_threshold: float = 1e-10) -> Tuple[mm.MOTAccumulator, Dict]:
        
        reference_frames = np_references[:, 0].max() if np_references.size > 0 else 0
        prediction_frames = np_predictions[:, 0].max() if np_predictions.size > 0 else 0

        num_frames = int(max(reference_frames, prediction_frames))

        acc = mm.MOTAccumulator(auto_id=True)

        for i in range(1, num_frames + 1):
            preds = np_predictions[np_predictions[:, 0] == i, 1:6]
            refs = np_references[np_references[:, 0] == i, 1:6]
            C = mm.distances.iou_matrix(
                refs[:, 1:], preds[:, 1:], max_iou=1 - iou_threshold
            )  # motmetrics expects iou association threshold to be smaller for stricter association
            acc.update(
                refs[:, 0].astype("int").tolist(), preds[:, 0].astype("int").tolist(), C
            )

        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics=["num_misses", "num_detections"]).to_dict()

        return acc, summary

    def calculate(
        self,
        predictions,
        references,
        ) -> Dict:
        """Returns the scores"""

        np_predictions, np_references = validate_inputs(predictions, references)

        acc, summary = self.motmetrics_compute(np_predictions, np_references, self.iou_threshold)

        df = events_to_df_map(acc.events)
        tr_ratios = track_ratios(df, obj_frequencies(df))
        unique_gt_ids = self.unique_obj_count(df)

        namemap = {"num_misses": "fn", "num_false_positives": "fp", "num_detections": "tp"}

        for key in list(summary.keys()):
            if key in namemap:
                summary[namemap[key]] = float(summary[key][0])
                summary.pop(key)
            else:
                summary[key] = float(summary[key][0])

        summary["unique_obj_count"] = unique_gt_ids

        for th in self.recognition_thresholds:
            recognized = self.recognition(tr_ratios, th)
            summary[f"mostly_tracked_count_{th}".replace(".", "_")] = int(recognized)

        return summary 

    @staticmethod
    def derive_scores(metrics_dict, recognition_thresholds) -> Dict:
        """
        calculates metrics based on raw metrics
        """

        metrics_dict["recall"] = metrics_dict["tp"] / (
            metrics_dict["tp"] + metrics_dict["fn"] + 1e-6
        )

        for th in recognition_thresholds:
            metrics_dict[f"mostly_tracked_score_{th}".replace(".", "_")] = metrics_dict[
                f"mostly_tracked_count_{th}".replace(".", "_")
            ] / (metrics_dict["unique_obj_count"] + 1e-6)

        return metrics_dict

    @staticmethod
    def recognition(track_ratios, th=0.5) -> int:
        """Number of objects tracked for at least 20 percent of lifespan."""
        return track_ratios[track_ratios >= th].count()

    @staticmethod
    def unique_obj_count(df) -> int:
        """Number of unique gt ids."""
        return df.full["OId"].dropna().unique().shape[0]