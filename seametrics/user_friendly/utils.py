from typing import Dict, List, Optional, Tuple

import motmetrics as mm
import numpy as np
import pandas as pd
from motmetrics.metrics import events_to_df_map, obj_frequencies, track_ratios

from seametrics.payload import Payload, Sequence


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
        np_predictions = np.array(predictions) if predictions else np.empty((0, 8))
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

    if np_predictions.ndim < 2 or np_predictions.shape[1] != 8:
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
        list: A list of lists in the format [frame number, object id, bb_left, bb_top, bb_width, bb_height, confidence, distance].

    """
    formated_predictions = []
    for frame_id, frame in enumerate(frames):
        for detection in frame:
            index = detection["index"]
            x, y, w, h = detection["bounding_box"]
            confidence = 1
            distance = detection["pos_polar"][0]
            formated_predictions.append(
                [frame_id + 1, index, x, y, w, h, confidence, distance]
            )

    return formated_predictions


def payload_to_uf_metrics(
    payload: Payload,
    model_name: str = None,
    filter_dict={"name": "area", "ranges": [("all", [0, 1e5**2])]},
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
        formated_references = get_formated_references(
            sequence[payload.gt_field_name], filter_dict
        )
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
        fps: int = 30,
        slice_length: Optional[int] = None,
    ):
        """
        Initialize the UserFriendly class.

        Parameters
        ----------
        iou_threshold : float
            The maximum intersection over union (IoU) threshold.
        recognition_thresholds : list
            A list of recognition thresholds to use.
        fps : int
            Frames per second of the video.
        slice_length : Optional[int]
            Length of each evaluation slice in seconds. If None, evaluates the full sequence.
        """
        self.iou_threshold = iou_threshold
        self.recognition_thresholds = recognition_thresholds
        self.fps = fps
        self.slice_length = slice_length

    def motmetrics_compute(
        self, np_predictions, np_references, iou_threshold: float = 1e-10
    ) -> List[Tuple[mm.MOTAccumulator, Dict]]:
        """
        Compute MOT metrics for each slice independently.

        Parameters
        ----------
        np_predictions : np.ndarray
            Array of predictions.
        np_references : np.ndarray
            Array of ground truth references.
        iou_threshold : float
            IoU threshold for association.

        Returns
        -------
        List of tuples, each containing a MOTAccumulator and metrics summary per slice.
        """
        acc_results = []

        if self.slice_length:

            np_predictions_sliced, np_references_sliced, frame_ranges = (
                self.slice_predictions_and_references(
                    np_predictions,
                    np_references,
                    slice_length=self.slice_length,
                    fps=self.fps,
                )
            )

            for preds_slice, refs_slice, (start_frame, end_frame) in zip(
                np_predictions_sliced, np_references_sliced, frame_ranges
            ):
                # Process each slice
                # Create a fresh accumulator for each slice
                acc = mm.MOTAccumulator(auto_id=True)

                num_frames = int(
                    max(
                        refs_slice[:, 0].max() if refs_slice.size > 0 else 0,
                        preds_slice[:, 0].max() if preds_slice.size > 0 else 0,
                    )
                )

                for i in range(start_frame, end_frame + 1):
                    preds = preds_slice[preds_slice[:, 0] == i, 1:6]
                    refs = refs_slice[refs_slice[:, 0] == i, 1:6]
                    _, counts = np.unique(refs[:, 0], return_counts=True)
                    if np.any(counts > 1):  # Only call function if duplicates exist
                        refs = self.resolve_duplicate_ids(refs)

                    # print(preds.shape, refs.shape)
                    # print(refs)
                    # print(preds)

                    C = mm.distances.iou_matrix(
                        refs[:, 1:], preds[:, 1:], max_iou=1 - iou_threshold
                    )

                    ref_ids = refs[:, 0].astype("int").tolist()
                    pred_ids = preds[:, 0].astype("int").tolist()

                    acc.update(ref_ids, pred_ids, C)

                # Compute and store results per slice
                mh = mm.metrics.create()
                summary = mh.compute(
                    acc, metrics=["num_misses", "num_detections"]
                ).to_dict()
                acc_results.append((acc, summary))  # Store each slice separately

        else:
            # Default version (no slicing) remains unchanged
            acc = mm.MOTAccumulator(auto_id=True)
            reference_frames = (
                np_references[:, 0].max() if np_references.size > 0 else 0
            )
            prediction_frames = (
                np_predictions[:, 0].max() if np_predictions.size > 0 else 0
            )
            num_frames = int(max(reference_frames, prediction_frames))

            for i in range(1, num_frames + 1):
                preds = np_predictions[np_predictions[:, 0] == i, 1:6]
                refs = np_references[np_references[:, 0] == i, 1:6]

                C = mm.distances.iou_matrix(
                    refs[:, 1:], preds[:, 1:], max_iou=1 - iou_threshold
                )

                acc.update(
                    refs[:, 0].astype("int").tolist(),
                    preds[:, 0].astype("int").tolist(),
                    C,
                )

            mh = mm.metrics.create()
            summary = mh.compute(
                acc, metrics=["num_misses", "num_detections"]
            ).to_dict()
            acc_results.append((acc, summary))

        return acc_results

    def calculate(
        self,
        predictions,
        references,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Returns a list of scores per slice separately"""

        np_predictions, np_references = validate_inputs(predictions, references)
        acc_summaries = self.motmetrics_compute(
            np_predictions, np_references, self.iou_threshold
        )

        summary_list = []
        my_dict_list = []
        distances_list = []

        for acc, summary in acc_summaries:
            hit_events = acc.mot_events[
                (acc.mot_events["Type"] == "MATCH")
                | (acc.mot_events["Type"] == "SWITCH")
            ]
            distances = self.extract_oid_distance_info(hit_events, predictions)

            df = events_to_df_map(acc.events)
            tr_ratios = track_ratios(df, obj_frequencies(df))
            tracked = df.noraw[df.noraw.Type != "MISS"]["OId"].value_counts()
            not_tracked = df.noraw[df.noraw.Type == "MISS"]["OId"].value_counts()

            unique_gt_ids = self.unique_obj_count(df)
            namemap = {
                "num_misses": "fn",
                "num_false_positives": "fp",
                "num_detections": "tp",
            }

            for key in list(summary.keys()):
                if key in namemap:
                    summary[namemap[key]] = float(summary[key][0])
                    summary.pop(key)
                else:
                    summary[key] = float(summary[key][0])

            summary["unique_obj_count"] = unique_gt_ids

            for th in self.recognition_thresholds:
                recognized = self.recognition(tr_ratios, th)
                summary[f"mostly_tracked_count_{th}".replace(".", "_")] = int(
                    recognized
                )

            my_dict = {
                "HIT_obj_count": tracked.to_dict(),
                "MISS_obj_count": not_tracked.to_dict(),
                "obj_count": obj_frequencies(df).to_dict(),
                "ratios": tr_ratios.to_dict(),
            }

            summary_list.append(summary)
            my_dict_list.append(my_dict)
            distances_list.append(distances)

        return summary_list, my_dict_list, distances_list

    @staticmethod
    def resolve_duplicate_ids(refs):
        refs = np.array(refs)
        existing_ids = set(refs[:, 0])
        unique_ids, counts = np.unique(refs[:, 0], return_counts=True)
        duplicates = unique_ids[counts > 1]

        for dup in duplicates:
            dup_indices = np.where(refs[:, 0] == dup)[0]
            new_id = max(existing_ids) + 1
            for idx in dup_indices[1:]:  # Keep the first, change the rest
                while new_id in existing_ids:
                    new_id += 1
                refs[idx, 0] = new_id
                existing_ids.add(new_id)

        return refs

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

    @staticmethod
    def extract_oid_distance_info(dist_df, predictions):
        """
        Extracts distance information for each OID based on its HId (FrameID) and returns a dictionary
        where keys are OIDs and values are lists of distances.

        Parameters:
            dist_df (pd.DataFrame): DataFrame containing columns ['OID', 'HId', ...] where HId is the FrameID.
            predictions (list of lists): List containing prediction data in the format:
                                        [frame number, object id, bb_left, bb_top, bb_width, bb_height, confidence, distance]

        Returns:
            dict: A dictionary where keys are OIDs and values are lists of distances.
        """
        # Convert prediction_data to DataFrame with column names
        prediction_df = pd.DataFrame(
            predictions,
            columns=["FrameId", "HId", "C1", "C2", "C3", "C4", "C5", "Distance"],
        )

        # Convert predictions to a DataFrame
        event_df = dist_df.reset_index()
        event_df["NextFrameId"] = event_df["FrameId"] + 1

        # event_df['HId'] = pd.to_numeric(event_df['HId'], errors='coerce')  # Convert to numeric
        prediction_df["HId"] = pd.to_numeric(
            prediction_df["HId"], errors="coerce"
        )  # Convert to numeric
        # Merge with prediction_df on NextFrameId and HId
        merged_df = event_df.merge(
            prediction_df,
            left_on=["NextFrameId", "HId"],
            right_on=[prediction_df.columns[0], prediction_df.columns[1]],
        )
        # print(merged_df)
        # Group by OId and extract last column (distance)
        grouped_distances = (
            merged_df.groupby("OId")[prediction_df.columns[-1]].apply(list).to_dict()
        )

        return grouped_distances

    @staticmethod
    def slice_predictions_and_references(
        np_predictions: np.ndarray,
        np_references: np.ndarray,
        slice_length: int,
        fps: int,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[Tuple[int, int]]]:
        """
        Slice np_predictions and np_references based on slice_length and fps,
        returning the start and end frame for each slice.

        Parameters
        ----------
        np_predictions : np.ndarray
            Array of predictions.
        np_references : np.ndarray
            Array of ground truth references.
        slice_length : int
            Length of each slice in seconds.
        fps : int
            Frames per second.

        Returns
        -------
        Tuple[List[np.ndarray], List[np.ndarray], List[Tuple[int, int]]]
            Lists of sliced np_predictions, np_references, and corresponding (start_frame, end_frame).
        """
        if slice_length <= 0 or fps <= 0:
            raise ValueError("slice_length and fps must be positive integers.")

        sliced_predictions = []
        sliced_references = []
        frame_ranges = []

        slice_frame_length = slice_length * fps  # Convert slice length to frames
        max_frame = np_references[:, 0].max() if np_references.size > 0 else 0

        for start_frame in range(1, int(max_frame) + 1, slice_frame_length):
            end_frame = (
                start_frame + slice_frame_length - 1
            )  # Fix: Ensure end_frame is inclusive

            refs_slice = np_references[
                (np_references[:, 0] >= start_frame)
                & (np_references[:, 0] <= end_frame)
            ]
            preds_slice = np_predictions[
                np.isin(np_predictions[:, 0], refs_slice[:, 0])
            ]

            sliced_predictions.append(preds_slice)
            sliced_references.append(refs_slice)
            frame_ranges.append((start_frame, end_frame))

        return sliced_predictions, sliced_references, frame_ranges
