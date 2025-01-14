import logging
from typing import Any, Dict, List, Literal

import fiftyone as fo
from deprecated import deprecated

from seametrics.payload.processor import PayloadProcessor


@deprecated(
    reason=(
        "\nUse the `PayloadProcessor(...).payload` instead."
        "\n`fo_to_payload` will be removed in the future."
        "\nhttps://github.com/SEA-AI/seametrics/tree/develop"
    )
)
def fo_to_payload(
    dataset: str,
    gt_field: str,
    models: List[str],
    tracking_mode: bool = False,
    sequence_list: List[str] = [],
    data_type: Literal["rgb", "thermal"] = "thermal",
    excluded_classes: List[str] = None,
    debug: bool = False,
) -> Dict:
    """
    Processes a dataset containing detections in frames and returns a formatted payload.

    Args:
        dataset: Name of the dataset containing detections.
        gt_field: Name of the ground-truth field in the dataset.
        models: List of model names used for detection.
        sequence_list: Optional list of sequence names to include.
            If no sequence list is provided, it will use all sequences on the dataset
        data_type: Optional data type. Will automatically choose the right fiftyone
            slice based on data type. Defaults to thermal data.
        img_size: Desired image size as a tuple (width, height).
        excluded_classes: Optional list of class names to exclude.

    Returns:
        A dictionary containing standard payload.
        If we have n frames per sequence and len(models)=l, it is of form:
        {
            'dataset': dataset name (as defined in input parameters)
            'models': list of model names (as defined in input parameters)
            'gt_field_name': ground truth field (as defined in input parameters)
            'sequences' : a dict containing a key for each sequence.
                'sequence_i' : {
                    'resolution': Resolution(height, width),
                    'model_0': [list_detections_frame_0, ..., list_detections_frame_n],
                    ...,
                    'model_l': [list_detections_frame_0, ..., list_detections_frame_n],
                    'gt_field_name': [list_gts_frame_0, ..., list_gts_frame_n]
                },
            'sequence_list': list of sequences (as defined in input parameters)
        }

        with list_detections_frame_i being a list of detections fo.Detection format.

    Raises:
        ValueError: If invalid input arguments are provided.
    """

    if debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    return PayloadProcessor(
        dataset_name=dataset,
        gt_field=gt_field,
        models=models,
        tracking_mode=tracking_mode,
        sequence_list=sequence_list,
        data_type=data_type,
        excluded_classes=excluded_classes,
    ).payload


def fo_upload(
    dataset_name: str,
    metrics: Dict[str, Dict],
    metric_name: str,
    description: Dict[str, Any] = None,
) -> None:
    """
    Upload detection metrics to a FiftyOne dataset. A new field is created
    for each model in the dataset. The field name is "polymetrics",
    and queryable ex: "{polymetrics}.{model_name}.{metric_name}.{area_range}.{'f1/precision/recall...'}".
    Note: this function is compatible with the Polymetrics tool.

    The metrics field should be a dictionary of the following form:
    {
        "model_name": {
            "overall": {
            "all": {"tp": ..., "fp": ..., "fn": ..., "f1": ...},
            ...  # more area ranges
            },
            "per_sequence": {
            "sequence_name": {
                "all": {...},
                ...  # more area ranges
            },
            ...  # more sequences
            }
        },
        ...  # more models
    }

    Args:
        dataset_name (str):
            The FiftyOne dataset view to update.
        metrics (dict):
            A dictionary containing metrics for multiple models and their sequences.
        metric_name (str):
            The base name for the metrics field.
        description (dict, optional):
            A dictionary of the metrics run configuration. This parameter is optional and defaults to `None` if not provided.

    Returns:
        None
    """
    # Load the dataset
    dataset = fo.load_dataset(dataset_name)
    # sanity-check to avoid overriding field
    dataset.add_sample_field("polymetrics", fo.DictField)

    if not metrics:
        logging.warning("metrics is empty. Skipping.")
        return

    for model_name, model_data in metrics.items():
        per_sequence = model_data.get("per_sequence", {})

        if not per_sequence:
            logging.warning(
                f"No per_sequence data found for model {model_name}. Skipping."
            )
            continue

        for sequence_name, sequence_metrics in per_sequence.items():
            sequence_view = dataset.match(fo.ViewField("sequence") == sequence_name)

            if len(sequence_view) == 0:
                logging.warning(f"Sequence {sequence_name} not found.")
                continue

            # Add metric to each sample
            for sample in sequence_view:
                if model_name not in sample["polymetrics"]:
                    sample["polymetrics"][model_name] = {}

                if metric_name not in sample["polymetrics"][model_name]:
                    sample["polymetrics"][model_name][metric_name] = {}

                # Update or add metric details
                sample["polymetrics"][model_name][metric_name].update(sequence_metrics)
                sample["polymetrics"][model_name][metric_name]["description"] = (
                    description
                )
                sample.save()
