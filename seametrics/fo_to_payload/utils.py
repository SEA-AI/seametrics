import fiftyone as fo
from fiftyone import ViewField as F
import typing
from tqdm import tqdm

def fo_to_payload(dataset: str, 
                  gt_field: str, 
                  models: typing.List[str], 
                  tracking_mode: bool = False,
                  sequence_list: typing.List[str] = [],
                  data_type: typing.Literal["rgb", "thermal"] = "thermal",
                  excluded_classes: typing.List[str] = None,
                  debug: bool = False) -> typing.Dict:
    """
    Processes a dataset containing detections in frames and returns a formatted payload.

    Args:
        dataset: Name of the dataset containing detections.
        gt_field: Name of the ground-truth field in the dataset.
        models: List of model names used for detection.
        sequence_list: Optional list of sequence names to include.
            If no sequence list is provided, it will use all sequences on the dataset
        data_type: Optional data type. Will automatically choose the right fiftyone slice based on data type.
            Defaults to thermal data.
        img_size: Desired image size as a tuple (width, height).
        excluded_classes: Optional list of class names to exclude.

    Returns:
        A dictionary containing standard payload.
        If we have n frames per sequence and len(models)=l, it is of form:
        {
            'dataset': dataset name (as defined in input parameters)
            'models': list of model names (as defined in input parameters)
            'gt_field_name': ground truth field (as defined in input parameters)
            'sequences' : a dict containing a key for each sequence. value for i-th sequence is:
                            'sequence_i' : {
                                'resolution': (height, width),
                                'model_0': [list_detections_frame_0, list_detections_frame_1, ..., list_detections_frame_n],
                                ...,
                                'model_l': [list_detections_frame_0, list_detections_frame_1, ..., list_detections_frame_n],
                                'gt_field_name': [list_gts_frame_0, list_gts_frame_1, ..., list_gts_frame_n]
                            },
            'sequence_list': list of sequences (as defined in input parameters)
        }

        with list_detections_frame_i being a list of detections in format of fo.Detection.

    Raises:
        ValueError: If invalid input arguments are provided.
    """
    if excluded_classes == None:
        excluded_classes = ['ALGAE','BRIDGE','HARBOUR','WATERTRACK','SHORELINE','SUN_REFLECTION','UNSUPERVISED','WATER','TRASH','OBJECT_REFLECTION','HORIZON']
    
    if debug:
        print(f"Processing dataset {dataset} with ground-truth field {gt_field} and models {models}.")
        print(f"Tracking mode: {tracking_mode}")
        print(f"Sequence list: {sequence_list}")
        print(f"Excluded classes: {excluded_classes}")

    if dataset not in fo.list_datasets():
        raise ValueError(f"Dataset {dataset} not found in FiftyOne.")
    
    # TODO: include automatic check: don't use tracking with sailing data as not supported
    if tracking_mode and data_type == "rgb":
        raise ValueError("Currently tracking-mode evaluation is not supported for RGB data.")

    fields = models + [gt_field]
    loaded_dataset = fo.load_dataset(dataset)

    output = {
        'dataset': dataset,
        'models': models,
        'gt_field_name': gt_field,
        'sequences': {},
    }

    if len(sequence_list) == 0:
        sequence_list = loaded_dataset.distinct("sequence")
        if debug:
            print(f"Using all sequences in dataset: {sequence_list}")

    # TODO: enable slice selection for sailing (RGB or thermal slices)
    # TODO: "frames.{gt_field}" & "frames[].{field}.detections" imply video dataset -> change to "{gt_field}" for image datasets
    # TODO: deal with varying image size in sailing data
    relevant_slices = get_datatype_slices(
        view=loaded_dataset,
        data_type=data_type
    )
    if debug:
        print(f"Using slice: {relevant_slices}")
    degrouped_dataset = loaded_dataset.select_group_slices(relevant_slices)
    gt_field_fo = get_field_name(degrouped_dataset, gt_field, is_gt=True)
    # Process each sequence in the sequence_list
    for sequence in tqdm(sequence_list):
        try:
            sequence_view = degrouped_dataset.match(F("sequence") == sequence) \
                            .filter_labels(gt_field_fo, ~F("label").is_in(excluded_classes), only_matches=False)
        except:
            raise ValueError(f"Sequence {sequence} not found in dataset {dataset}.")

        try:           
            if tracking_mode:
                mux = sequence_view.values(f"frames[].mux")
                mux = [item if item else [] for item in mux]
        except:
            print(f'Memory error on sequence {sequence}.')
            continue          

        output['sequences'][sequence] = {}
        output['sequences'][sequence]["resolution"] = get_resolution(sequence_view)
        for field in fields:
            pred_field_fo = get_field_name(degrouped_dataset, field, is_gt=False)
            predictions = sequence_view.values(f"{pred_field_fo}.detections") # length: number of images/frames in sequence
            if tracking_mode:
                output['sequences'][sequence][field] = [prediction for mux_item,prediction in zip(mux, predictions) if len(mux_item) != 0]
            else:
                output['sequences'][sequence][field] = predictions
            #replace None with empty list
            output['sequences'][sequence][field] = [[] if x == None else x for x in output['sequences'][sequence][field]]
    output['sequence_list'] = list(output['sequences'].keys())
    return output

def get_resolution(view: fo.DatasetView) -> typing.Tuple[int, int]:
    """Get resolution from fiftyone view.
    Warning: This assumes that all samples in this view have the same size.

    Args:
        view (fo.DatasetView): Fiftyone Dataset containing single sequence (or multiple but with equal img resolution)

    Returns:
        typing.Tuple[int, int]: (height, width) of image in pixel
    """
    # TODO: maybe include check to assert that shape really always is the same throughout view?
    if view.media_type == 'video':
        return (view.first().metadata.frame_height, \
                view.first().metadata.frame_width)
    elif view.media_type == 'image':
        return (view.first().metadata.height, \
                view.first().metadata.width)
    else:
        raise ValueError(f"Unsupported media type: {view.media_type}")

def get_datatype_slices(view: fo.DatasetView,
                        data_type: typing.Literal["rgb", "thermal"]) -> fo.DatasetView:
    """Return view containing relevant group slices.

    Args:
        view (fo.DatasetView): Fiftyone Dataset
        data_type (typing.Literal["rgb", "thermal"]): which data format should be selected

    Returns:
        fo.DatasetView: fiftyone dataset view including relevant group slices
    """
    existing_slices = view.group_slices
    chosen_slices = []

    if data_type == "thermal":
        if "thermal_wide" in existing_slices:
            chosen_slices.append("thermal_wide")
        if "thermal_right" in existing_slices:
            chosen_slices.append("thermal_right")
        if "thermal_left" in existing_slices:
            chosen_slices.append("thermal_left")
    elif data_type == "rgb":
        if "rgb" in existing_slices:
            chosen_slices.append("rgb")
        elif ("rgb_wide" or "rgb_narrow") in existing_slices:
            raise ValueError("RGB data currently cannot be evaluated in video data.")
    else:
        raise ValueError(f"Found no matching data slice for datatype: {data_type}.")
    
    if len(chosen_slices) == 0:
        raise ValueError(f"This dataset has no slice corresponding to datatype: {data_type}")

    return chosen_slices
    
def get_field_name(view: fo.DatasetView,
                   field_name: str,
                   is_gt: bool = False) -> str:
    """
    Args:
        view (fo.DatasetView): Ungrouped dataset view
        field_name (str): Fiftyone field name.
            You can use dot notation (embedded.field.name).

    Returns:
        list: List of values.
    """

    if view.media_type == 'video':
        if is_gt:
            return f"frames.{field_name}"
        else:
            return f"frames[].{field_name}"
    elif view.media_type == 'image':
        return field_name
    else:
        raise ValueError(f"Unsupported media type: {view.media_type}")