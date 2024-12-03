import motmetrics as mm
import numpy as np
from motmetrics.metrics import events_to_df_map, obj_frequencies, track_ratios


def recognition(track_ratios, th=0.5):
    """Number of objects tracked for at least 20 percent of lifespan."""
    return track_ratios[track_ratios >= th].count()


def num_gt_ids(df):
    """Number of unique gt ids."""
    return df.full["OId"].dropna().unique().shape[0]


def calculate(
    predictions,
    references,
    max_iou: float = 0.5,
    recognition_thresholds: list = [0.3, 0.5, 0.8],
):
    """Returns the scores"""

    try:
        np_predictions = np.array(predictions)
    except:
        raise ValueError(
            "The predictions should be a list of np.arrays in the format [frame number, object id, bb_left, bb_top, bb_width, bb_height, confidence]"
        )

    try:
        np_references = np.array(references)
    except:
        raise ValueError(
            "The references should be a list of np.arrays in the format [frame number, object id, bb_left, bb_top, bb_width, bb_height]"
        )

    if np_predictions.shape[1] != 7:
        raise ValueError(
            "The predictions should be a list of np.arrays in the format [frame number, object id, bb_left, bb_top, bb_width, bb_height, confidence]"
        )
    if np_references.shape[1] != 6:
        raise ValueError(
            "The references should be a list of np.arrays in the format [frame number, object id, bb_left, bb_top, bb_width, bb_height]"
        )

    if np_predictions[:, 0].min() <= 0:
        raise ValueError(
            "The frame number in the predictions should be a positive integer"
        )
    if np_references[:, 0].min() <= 0:
        raise ValueError(
            "The frame number in the references should be a positive integer"
        )

    num_frames = int(max(np_references[:, 0].max(), np_predictions[:, 0].max()))

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
        acc, metrics=["num_misses", "num_false_positives", "num_detections"]
    ).to_dict()

    df = events_to_df_map(acc.events)
    tr_ratios = track_ratios(df, obj_frequencies(df))
    unique_gt_ids = num_gt_ids(df)

    namemap = {"num_misses": "fn", "num_false_positives": "fp", "num_detections": "tp"}

    for key in list(summary.keys()):
        if key in namemap:
            summary[namemap[key]] = float(summary[key][0])
            summary.pop(key)
        else:
            summary[key] = float(summary[key][0])

    summary["num_gt_ids"] = unique_gt_ids

    for th in recognition_thresholds:
        recognized = recognition(tr_ratios, th)
        summary[f"recognized_{th}"] = int(recognized)

    return summary


def build_metrics_template(models, filters):
    metrics_dict = {}
    for model in models:
        metrics_dict[model] = {}
        metrics_dict[model]["all"] = {}
        for filter, filter_ranges in filters.items():
            metrics_dict[model][filter] = {}
            for filter_range in filter_ranges:
                filter_range_name = filter_range[0]
                metrics_dict[model][filter][filter_range_name] = {}
    return metrics_dict


def calculate_from_payload(
    payload: dict,
    max_iou: float = 0.5,
    filters={},
    recognition_thresholds=[0.3, 0.5, 0.8],
    debug: bool = False,
):

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

    metrics_per_sequence = {}
    metrics_global = build_metrics_template(models, filters)

    for sequence in sequence_list:
        metrics_per_sequence[sequence] = {}
        frames = payload["sequences"][sequence][gt_field_name]

        all_formated_references = {"all": []}
        for filter, filter_ranges in filters.items():
            all_formated_references[filter] = {}
            for filter_range in filter_ranges:
                filter_range_name = filter_range[0]
                all_formated_references[filter][filter_range_name] = []

        for frame_id, frame in enumerate(frames):
            for detection in frame:
                index = detection["index"]
                x, y, w, h = detection["bounding_box"]
                all_formated_references["all"].append([frame_id + 1, index, x, y, w, h])

                for filter, filter_ranges in filters.items():
                    filter_value = detection[filter]
                    for filter_range in filter_ranges:
                        filter_range_name, filter_range_limits = (
                            filter_range[0],
                            filter_range[1],
                        )
                        if (
                            filter_value >= filter_range_limits[0]
                            and filter_value <= filter_range_limits[1]
                        ):
                            all_formated_references[filter][filter_range_name].append(
                                [frame_id + 1, index, x, y, w, h]
                            )

        metrics_per_sequence[sequence] = build_metrics_template(models, filters)

        for model in models:
            frames = payload["sequences"][sequence][model]
            formated_predictions = []

            for frame_id, frame in enumerate(frames):
                for detection in frame:
                    index = detection["index"]
                    x, y, w, h = detection["bounding_box"]
                    confidence = 1
                    formated_predictions.append(
                        [frame_id + 1, index, x, y, w, h, confidence]
                    )

            if debug:
                print("sequence/model: ", sequence, model)
                print("formated_predictions: ", formated_predictions)
                print("formated_references: ", all_formated_references)

            if len(formated_predictions) == 0:
                metrics_per_sequence[sequence][model] = "Model had no predictions."
            elif len(all_formated_references["all"]) == 0:
                metrics_per_sequence[sequence][model] = "No ground truth."

            else:

                sequence_metrics = calculate(
                    formated_predictions,
                    all_formated_references["all"],
                    max_iou=max_iou,
                    recognition_thresholds=recognition_thresholds,
                )
                sequence_metrics = realize_metrics(
                    sequence_metrics, recognition_thresholds
                )
                metrics_per_sequence[sequence][model]["all"] = sequence_metrics

                metrics_global[model]["all"] = sum_dicts(
                    metrics_global[model]["all"], sequence_metrics
                )
                metrics_global[model]["all"] = realize_metrics(
                    metrics_global[model]["all"], recognition_thresholds
                )

                for filter, filter_ranges in filters.items():

                    for filter_range in filter_ranges:

                        filter_range_name = filter_range[0]
                        sequence_metrics = calculate(
                            formated_predictions,
                            all_formated_references[filter][filter_range_name],
                            max_iou=max_iou,
                            recognition_thresholds=recognition_thresholds,
                        )
                        sequence_metrics = realize_metrics(
                            sequence_metrics, recognition_thresholds
                        )
                        metrics_per_sequence[sequence][model][filter][
                            filter_range_name
                        ] = sequence_metrics

                        metrics_global[model][filter][filter_range_name] = sum_dicts(
                            metrics_global[model][filter][filter_range_name],
                            sequence_metrics,
                        )
                        metrics_global[model][filter][filter_range_name] = (
                            realize_metrics(
                                metrics_global[model][filter][filter_range_name],
                                recognition_thresholds,
                            )
                        )

    output = {"global": metrics_global, "per_sequence": metrics_per_sequence}

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

    if metrics_dict["tp"] + metrics_dict["fp"] > 0:
        metrics_dict["precision"] = metrics_dict["tp"] / (
            metrics_dict["tp"] + metrics_dict["fp"]
        )
    else:
        metrics_dict["precision"] = np.nan

    if metrics_dict["tp"] + metrics_dict["fn"] > 0:
        metrics_dict["recall"] = metrics_dict["tp"] / (
            metrics_dict["tp"] + metrics_dict["fn"]
        )
    else:
        metrics_dict["recall"] = np.nan

    metrics_dict["f1"] = (
        2
        * metrics_dict["precision"]
        * metrics_dict["recall"]
        / (metrics_dict["precision"] + metrics_dict["recall"] + 1e-6)
    )

    for th in recognition_thresholds:
        metrics_dict[f"recognition_{th}"] = (
            metrics_dict[f"recognized_{th}"] / metrics_dict["num_gt_ids"]
        )

    return metrics_dict
