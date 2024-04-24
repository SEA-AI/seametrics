import logging
from deprecated import deprecated
from typing import List, Dict, Literal
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
