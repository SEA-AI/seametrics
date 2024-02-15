import fiftyone as fo
from fiftyone import ViewField as F
import typing
from tqdm import tqdm

def fo_to_payload(dataset: str, 
                gt_field: str, 
                models: typing.List[str], 
                tracking_mode: bool = False,
                sequence_list: typing.List[str] = [],
                img_size: typing.Tuple[int, int] = (640, 512),
                excluded_classes: typing.List[str] = None,
                debug: bool = False
                ) -> typing.Dict:
    """
    Processes a dataset containing detections in frames and returns a formatted payload.

    Args:
        dataset: Name of the dataset containing detections.
        gt_field: Name of the ground-truth field in the dataset.
        models: List of model names used for detection.
        sequence_list: Optional list of sequence names to include. If no sequence list is provided, it will use all sequences on the dataset
        img_size: Desired image size as a tuple (width, height).
        excluded_classes: Optional list of class names to exclude.

    Returns:
        A dictionary containing standard payload.

    Raises:
        ValueError: If invalid input arguments are provided.
    """
    # TODO: include automatic check: don't use tracking with sailing data as not supported
    if debug:
        print(f"Processing dataset {dataset} with ground-truth field {gt_field} and models {models}.")
        print(f"Tracking mode: {tracking_mode}")
        print(f"Sequence list: {sequence_list}")
        print(f"Image size: {img_size}")
        print(f"Excluded classes: {excluded_classes}")

    if excluded_classes == None:
        excluded_classes = ['ALGAE','BRIDGE','HARBOUR','WATERTRACK','SHORELINE','SUN_REFLECTION','UNSUPERVISED','WATER','TRASH','OBJECT_REFLECTION','HORIZON']

    if dataset not in fo.list_datasets():
        raise ValueError(f"Dataset {dataset} not found in FiftyOne.")

    loaded_dataset = fo.load_dataset(dataset)
    fields = models + [gt_field]

    output = {
        'dataset': dataset,
        'models': models,
        'gt_field_name': gt_field,
        'img_size': img_size,
        'sequences': {},
        'sequence_list': sequence_list
    }
    if len(sequence_list) == 0:
        sequence_list = loaded_dataset.distinct("sequence")
        if debug:
            print(f"Using all sequences in dataset: {sequence_list}")

    # TODO: enable slice selection for sailing (RGB or thermal slices)
    # TODO: "frames.{gt_field}" & "frames[].{field}.detections" implie video dataset -> change to "{gt_field}" for image datasets
    # Process each sequence in the sequence_list
    for sequence in tqdm(sequence_list):
        try:
            sequence_view = loaded_dataset.select_group_slices(["thermal_wide"]) \
                        .match(F("sequence") == sequence) \
                        .filter_labels(f"frames.{gt_field}", ~F("label").is_in(excluded_classes), only_matches=False)
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
        for field in fields:
            predictions = sequence_view.values(f"frames[].{field}.detections")
            if tracking_mode:
                output['sequences'][sequence][field] = [prediction for mux_item,prediction in zip(mux, predictions) if len(mux_item) != 0]
            else:
                output['sequences'][sequence][field] = predictions
            #replace None with empty list
            output['sequences'][sequence][field] = [[] if x == None else x for x in output['sequences'][sequence][field]]
    return output