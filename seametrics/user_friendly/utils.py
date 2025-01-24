import motmetrics as mm
import numpy as np
from motmetrics.metrics import events_to_df_map, obj_frequencies, track_ratios


def recognition(track_ratios, th=0.5):
    """Number of objects tracked for at least 20 percent of lifespan."""
    return track_ratios[track_ratios >= th].count()


def unique_obj_count(df):
    """Number of unique gt ids."""
    return df.full["OId"].dropna().unique().shape[0]

def trasform_inputs(predictions, references):

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

    if (
        np_predictions.ndim < 2 or np_predictions.shape[1] != 7
    ):
        raise ValueError(
            "The predictions should be a 2D array with 7 columns in the format [frame number, object id, bb_left, bb_top, bb_width, bb_height, confidence]"
        )

    if np_references.ndim < 2 or np_references.shape[1] != 6:
        raise ValueError(
            "The references should be a 2D array with 6 columns in the format [frame number, object id, bb_left, bb_top, bb_width, bb_height]"
        )

    if np_predictions.size > 0:
        if np_predictions[:, 0].min() <= 0 :
            raise ValueError(
                "The frame number in the predictions should be a positive integer"
        )
    if np_references.size > 0:
        if np_references[:, 0].min() <= 0:
            raise ValueError(
                "The frame number in the references should be a positive integer"
        )

    return np_predictions, np_references

def calculate(
    predictions,
    references,
    max_iou: float = 0.5,
    recognition_thresholds: list = [0.3, 0.5, 0.8],
):
    """Returns the scores"""

    np_predictions, np_references = trasform_inputs(predictions, references)
    
    reference_frames = np_references[:, 0].max() if np_references.size > 0 else 0
    prediction_frames = np_predictions[:, 0].max() if np_predictions.size > 0 else 0

    print(len(predictions), len(references))
    
    num_frames = int(max(reference_frames, prediction_frames))

    acc = mm.MOTAccumulator(auto_id=True)

    for i in range(1, num_frames + 1):
        preds = np_predictions[np_predictions[:, 0] == i, 1:6]
        refs = np_references[np_references[:, 0] == i, 1:6]
        C = mm.distances.iou_matrix(
            refs[:, 1:], preds[:, 1:], max_iou=1 - max_iou
        )  # motmetrics expects iou association threshold to be smaller for stricter association
        acc.update(
            refs[:, 0].astype("int").tolist(), preds[:, 0].astype("int").tolist(), C
        )

    mh = mm.metrics.create()
    summary = mh.compute(
        acc, metrics=["num_misses", "num_detections"]
    ).to_dict()

    df = events_to_df_map(acc.events)
    tr_ratios = track_ratios(df, obj_frequencies(df))
    unique_gt_ids = unique_obj_count(df)

    namemap = {"num_misses": "fn", "num_false_positives": "fp", "num_detections": "tp"}

    for key in list(summary.keys()):
        if key in namemap:
            summary[namemap[key]] = float(summary[key][0])
            summary.pop(key)
        else:
            summary[key] = float(summary[key][0])

    summary["unique_obj_count"] = unique_gt_ids

    for th in recognition_thresholds:
        recognized = recognition(tr_ratios, th)
        summary[f"mostly_tracked_count_{th}"] = int(recognized)

    return summary


def build_metrics_template(filter):
    """builds the metrics template"""
    metrics_dict = {}
    for filter_range in filter:
        filter_range_name = filter_range[0]
        metrics_dict[filter_range_name] = {}
    return metrics_dict

def get_formated_references(frames, filter):
    """formats the references for the calculate_from_payload function, based on the filter and its ranges"""
    
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

def get_formated_predictions(frames):
    """formats the predictions for the calculate_from_payload function"""
    formated_predictions = []
    for frame_id, frame in enumerate(frames):
        for detection in frame:
            index = detection["index"]
            x, y, w, h = detection["bounding_box"]
            confidence = 1
            formated_predictions.append(
                [frame_id + 1, index, x, y, w, h, confidence]
            )

    return formated_predictions

def calculate_from_payload(payload: dict,
                        max_iou: float = 0.5, 
                        filter={"name": "area", 
                                "ranges": [("all", [0, 1e5**2])]},
                        recognition_thresholds=[0.3, 0.5, 0.8], 
                        debug: bool = False):
    """
    Filter in the form of:
    {
    "name": "area"
    "ranges": [("all", [0, 1e5**2]), ("small", [0**2, 6**2]), ("medium", [6**2, 12**2]), ("large", [12**2, 1e5**2])]
    }

    Receives a payload and returns the metrics
    """
    
    filter_name = filter["name"]
    filter_ranges = filter["ranges"]
    output = {}

    if not isinstance(payload, dict):
        try:
            payload = payload.to_dict()
        except Exception as e:
            raise ValueError(
                "The payload should be a dictionary or a compatible object"
            ) from e

    gt_field_name = payload["gt_field_name"]
    models = payload["models"]
    sequence_list = payload["sequence_list"]

    if debug:
        print("gt_field_name: ", gt_field_name)
        print("models: ", models)
        print("sequence_list: ", sequence_list)
    
    for model in models:

        metrics_overall = build_metrics_template(filter["ranges"])
        metrics_per_sequence = {}
        for sequence in sequence_list:

            metrics_per_sequence[sequence] = build_metrics_template(filter["ranges"])
            print(payload["sequences"])
            frames = payload["sequences"][sequence][gt_field_name]
            formated_references = get_formated_references(frames, filter)
            print(formated_references)


            frames = payload["sequences"][sequence][model]
            formated_predictions = get_formated_predictions(frames)
            print(formated_predictions)

            for filter_range in filter_ranges:
                
                filter_range_name = filter_range[0]

                sequence_metrics = calculate(
                    formated_predictions,
                    formated_references[filter_range_name],
                    max_iou=max_iou,
                    recognition_thresholds=recognition_thresholds,
                )

                metrics_per_sequence[sequence][filter_range_name] = realize_metrics(
                    sequence_metrics,
                    recognition_thresholds,
                )

                metrics_overall[filter_range_name] = sum_dicts(
                    metrics_overall[filter_range_name],
                    metrics_per_sequence[sequence][filter_range_name],
                )

                metrics_overall[filter_range_name] = realize_metrics(
                    metrics_overall[filter_range_name],
                    recognition_thresholds,
                )
                
        output[model] = {
            "per_sequence": metrics_per_sequence,
            "overall": metrics_overall,
        }

    return output
            


def sum_dicts(dict1, dict2):
    """
    Recursively sums the numerical values in two nested dictionaries.
    """
    result = {}
    for key in dict1.keys() | dict2.keys():  # Union of keys from both dictionaries
        val1 = dict1.get(key, 0)
        val2 = dict2.get(key, 0)
        if isinstance(val1, dict) and isinstance(val2, dict):
            # If both values are dictionaries, recursively sum them
            result[key] = sum_dicts(val1, val2)
        elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            # If both are numbers, sum them
            result[key] = val1 + val2
        else:
            # If only one dictionary has the key, take the non-zero value
            result[key] = val1 if val1 != 0 else val2
    return result


def realize_metrics(metrics_dict, recognition_thresholds):
    """
    calculates metrics based on raw metrics
    """

    metrics_dict["recall"] = metrics_dict["tp"] / (
        metrics_dict["tp"] + metrics_dict["fn"] + 1e-6
    )

    for th in recognition_thresholds:
        metrics_dict[f"mostly_tracked_score_{th}"] = (
            metrics_dict[f"mostly_tracked_count_{th}"]
            / (metrics_dict["unique_obj_count"]+1e-6)
        )

    return metrics_dict
