import os
import contextlib
import io
from tqdm import tqdm
import numpy as np
import pandas as pd
import fiftyone as fo
from fiftyone import ViewField as F
from tracking import TrackingMetrics

# helper functions

def prepare_data_for_det_metrics(gt_bboxes_per_frame,
                                 gt_track_ids_per_frame,
                                 dt_bboxes_per_frame,
                                 dt_track_ids_per_frame,
                                 dt_scores_per_frame,
                                 img_w: int = 640,
                                 img_h: int = 512):
    """
    Returns
    -------
    target, preds: tuple of list of dicts
        Each dict has keys "boxes", "labels", "scores" (scores is only in preds)
    """

    def _to_tracker_format(gt_bboxes_per_frame, gt_track_ids_per_frame,
                      dt_bboxes_per_frame, dt_track_ids_per_frame, dt_scores_per_frame,
                      img_w: int = 640, img_h: int = 512):
        """Converts a list of frames with detections (bboxes) to numpy format."""

        # Tracker format <frame number>, <object id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <confidence>, <x>, <y>, <z>
        # put to numpy format
        target = []
        preds = []

        for idx, (bbox, track_id) in enumerate(zip(gt_bboxes_per_frame, gt_track_ids_per_frame)):
            if bbox is not  None:
                for bb, t_id in zip(bbox, track_id):
                    denormalized_box = box_convert(
                    box_denormalize(np.array(bb), img_w, img_h),
                    in_fmt="xywh",
                    out_fmt="xyxy"
                    )
                    target.append([idx+1, t_id, denormalized_box[0], denormalized_box[1], denormalized_box[2], denormalized_box[3], 1, -1, -1, -1])

        for idx, (bbox, track_id, score) in enumerate(zip(dt_bboxes_per_frame, dt_track_ids_per_frame, dt_scores_per_frame)):
            if bbox is not None:
                for bb, t_id, s in zip(bbox, track_id, score):
                    denormalized_box = box_convert(
                    box_denormalize(np.array(bb), img_w, img_h),
                    in_fmt="xywh",
                    out_fmt="xyxy"
                    )
                    preds.append([idx+1, t_id, denormalized_box[0], denormalized_box[1], denormalized_box[2], denormalized_box[3], s, -1, -1, -1])
                   
        return np.array(target), np.array(preds)


    def _validate_arrays(data, data_type: str):
        if data is None or len(data) == 0:
            data = [data]
        if data_type in ["bbox", "mask"]:
            if any([(_not_falsy(item) and not isinstance(item[0], (tuple, list, np.ndarray)))
                    for item in data]):
                data = [data]
        elif data_type in ["score", "label"]:
            if any([(_not_falsy(item) and not isinstance(item, (tuple, list, np.ndarray)))
                    for item in data]):
                data = [data]
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
        data = [np.array(x) if x is not None else np.array([])
                for x in data]
        return data

    def _not_falsy(x):
        if x is None:
            return False
        if isinstance(x, (list, tuple)) and len(x) == 0:
            return False
        if isinstance(x, np.ndarray) and x.size == 0:
            return False
        return True

    # gt_bboxes_per_frame = _validate_arrays(
    #     gt_bboxes_per_frame, data_type="bbox")
    # gt_labels_per_frame = _validate_arrays(
    #     gt_labels_per_frame, data_type="label")
    # dt_bboxes_per_frame = _validate_arrays(
    #     dt_bboxes_per_frame, data_type="bbox")
    # dt_labels_per_frame = _validate_arrays(
    #     dt_labels_per_frame, data_type="label")
    # dt_scores_per_frame = _validate_arrays(
    #     dt_scores_per_frame, data_type="score")

    # assert len(gt_bboxes_per_frame) == len(
    #     dt_bboxes_per_frame), "Number of frames in GT and prediction do not match" + \
    #     f" ({len(gt_bboxes_per_frame)} vs {len(dt_bboxes_per_frame)})"

    # if all([item is None for sublist in dt_scores_per_frame for item in sublist]):
    #     # print("All scores are None, setting them to 1.0")
    #     dt_scores_per_frame = [[1.0] * len(x) for x in dt_scores_per_frame]

    target, preds = _to_tracker_format(
        gt_bboxes_per_frame, gt_track_ids_per_frame,
        dt_bboxes_per_frame, dt_track_ids_per_frame, dt_scores_per_frame)



    return target, preds

def get_relevant_fields(view: fo.DatasetView,
                        fields: list,  # fiftyone field names
                        ):
    """
    Returns a view with only the relevant fields to prevent memory issues.

    Parameters
    ----------
    view: fo.DatasetView
        Dataset view
    fields: list
        List of fiftyone field names. You can use dot notation (embedded.field.name).

    Returns
    -------
    fo.DatasetView
        Dataset view with only the relevant fields.
    """

    if view.media_type == 'video':
        return view.select_fields([f"frames.{f}" if view.has_frame_field(f) else f
                                   for f in fields])
    else:
        raise ValueError(f"Unsupported media type: {view.media_type}")

def get_values(view: fo.DatasetView,
               field_name: str,  # fiftyone field name
               ):
    """
    Parameters
    ----------
    view: fo.DatasetView
        Dataset view
    field_name: str
        Fiftyone field name. You can use dot notation (embedded.field.name).

    Returns
    -------
    list
        List of values.
    """

    if view.media_type == 'video':
        return view.values(f"frames[].{field_name}")
    elif view.media_type == 'image':
        return view.values(field_name)
    else:
        raise ValueError(f"Unsupported media type: {view.media_type}")

def compute_metrics(view: fo.DatasetView, # view
                    gt_field: str,  # fiftyone field name
                    pred_field: str  # fiftyone field name
                    ):  
    """Computes metrics for a given sequence view."""
    
    view = get_relevant_fields(view, [gt_field, pred_field, "mux"])
    gt_bboxes_per_frame = get_values(view,
                                     f"{gt_field}.detections.bounding_box")
    gt_track_ids_per_frame = get_values(view,
                                     f"{gt_field}.detections.index")
    dt_bboxes_per_frame = get_values(view,
                                     f"{pred_field}.detections.bounding_box")
    dt_scores_per_frame = get_values(view,
                                     f"{pred_field}.detections.confidence")
    dt_track_ids_per_frame = get_values(view,
                                     f"{pred_field}.detections.index")
    mux = get_values(view, "mux")
    

    gt_bboxes_per_frame = [bboxes for (mux_item,bboxes) in zip(mux, gt_bboxes_per_frame) if mux_item]
    gt_track_ids_per_frame = [track_ids for (mux_item, track_ids)  in zip(mux, gt_track_ids_per_frame) if mux_item]
    dt_bboxes_per_frame = [bboxes for (mux_item,bboxes) in zip(mux, dt_bboxes_per_frame) if mux_item]
    dt_scores_per_frame = [scores for (mux_item,scores) in zip(mux, dt_scores_per_frame) if mux_item]
    dt_track_ids_per_frame = [track_ids for (mux_item,track_ids) in zip(mux, dt_track_ids_per_frame) if mux_item]

    target, preds = prepare_data_for_det_metrics(
        gt_bboxes_per_frame, gt_track_ids_per_frame,
        dt_bboxes_per_frame, dt_track_ids_per_frame,  dt_scores_per_frame)

    # free memory
    del gt_bboxes_per_frame, gt_track_ids_per_frame, mux
    del dt_bboxes_per_frame, dt_scores_per_frame, dt_track_ids_per_frame

    return target, preds

def sequence_results_to_df(sequence_results):
    # save to pandas dataframe
    columns = ["sequence", "area_range_lbl", "area_range", "iou_threshold", "max_dets",
               "tp", "fp", "fn", "duplicates", "precision", "recall", "f1", "support",
               "fpi", "n_imgs"]
    df = pd.DataFrame(columns=columns)

    for seq_name, results in sequence_results.items():
        for area_range_lbl, metric in results['metrics'].items():
            # print(f"{seq_name} - {area_range_lbl}: {metric}")
            df.loc[len(df)] = {
                "sequence": seq_name,
                "area_range_lbl": area_range_lbl,
                "area_range": metric["range"],
                "iou_threshold": float(metric["iouThr"]),
                "max_dets": metric["maxDets"],
                "tp": metric["tp"],
                "fp": metric["fp"],
                "fn": metric["fn"],
                "duplicates": metric["duplicates"],
                "precision": metric["precision"],
                "recall": metric["recall"],
                "f1": metric["f1"],
                "support": metric["support"],
                "fpi": metric["fpi"],
                "n_imgs": metric["nImgs"],
            }

    return df

def compute_and_save_sequence_metrics(
    csv_dirpath: str,
    view: fo.DatasetView,
    gt_field: str,  # fiftyone field name
    pred_field: str,  # fiftyone field name
    metric_fn: callable,  # torchmetrics metric
    metric_kwargs: dict,  # kwargs for metric_fn
    csv_suffix: str = None,
    debug: bool = False,
    name_separator: str = "__",
    ):
    csv_name = name_separator.join(
        [view.dataset_name, gt_field, pred_field, metric_fn.__name__])
    csv_name = name_separator.join(
        [csv_name, csv_suffix]) if csv_suffix else csv_name
    csv_name += ".csv"
    csv_path = os.path.join(csv_dirpath, csv_name)
    print(f"Saving metrics to {csv_path}")

    view = get_relevant_fields(view, [gt_field, pred_field, 'sequence'])

    sequence_results = {}
    sequence_names = view.distinct("sequence")
    for sequence_name in tqdm(sequence_names):
        with contextlib.redirect_stdout(io.StringIO()) as f:
            print(sequence_name)
            sequence_view = view.match(F("sequence") == sequence_name)
            sequence_results[sequence_name] = compute_metrics(
                view=sequence_view,
                gt_field=gt_field,
                pred_field=pred_field,
                metric_fn=metric_fn,
                metric_kwargs=metric_kwargs,
            )

        if debug:
            print(f.getvalue())

    # create csv dir if not exists
    if not os.path.exists(csv_dirpath):
        os.makedirs(csv_dirpath)
    df = sequence_results_to_df(sequence_results)
    df.to_csv(csv_path, index=False)

def compute_metrics_by_sequence(
    view: fo.DatasetView,
    gt_field: str,  # fiftyone field name
    pred_field: str,  # fiftyone field name
    metric_fn: callable,  # torchmetrics metric
    metric_kwargs: dict,  # kwargs for metric_fn
    sequence_list: list = [], # list of sequence names
    ):

    sequence_results = {}
    
    metric = metric_fn(**metric_kwargs)
    if sequence_list is None:
        sequence_list = get_relevant_fields(view, [gt_field, pred_field, 'sequence']).distinct("sequence")
    for sequence_name in sequence_list:

        sequence_view = view.match(F("sequence") == sequence_name)
        sequence_results[sequence_name] = compute_metrics(
            view=sequence_view,
            gt_field=gt_field,
            pred_field=pred_field
        )
    for sequence in sequence_results.keys():
        try:
            
            metric.update(sequence_results[sequence][0], sequence_results[sequence][1], sequence)
        except:
            print(f"Sequence {sequence} failed")
        
    return metric

def compute_sizes(view: fo.DatasetView,
                    gt_field: str,  # fiftyone field name
                    img_w: int = 640,
                    img_h: int = 512):
        """Computes sizes for a given sequence view."""
        
        view = get_relevant_fields(view, [gt_field, "mux"])
        gt_bboxes_per_frame = get_values(view,
                                         f"{gt_field}.detections.bounding_box")
        gt_track_ids_per_frame = get_values(view,
                                            f"{gt_field}.detections.index")
        
        #mux = get_values(view, "mux")
        # gt_bboxes_per_frame = [bboxes for (mux_item,bboxes) in zip(mux, gt_bboxes_per_frame) if mux_item]
        # gt_track_ids_per_frame = [track_ids for (mux_item, track_ids)  in zip(mux, gt_track_ids_per_frame) if mux_item]
        b = [(bboxes, t_ids) for (bboxes,t_ids) in zip(gt_bboxes_per_frame, gt_track_ids_per_frame) if bboxes is not None and t_ids is not None]
        gt_bboxes_per_frame = [bboxes for (bboxes,_) in b]
        gt_track_ids_per_frame = [t_ids for (_,t_ids)  in b]
        objects = []
        for idx, (bbox,t_id) in enumerate(zip(gt_bboxes_per_frame, gt_track_ids_per_frame)):
            if bbox is not None:
                for (bb,track_id) in zip(bbox, t_id):
                    denormalized_box = box_denormalize(np.array(bb), img_w, img_h)
                    objects.append([idx, track_id, denormalized_box[2]*denormalized_box[3]])
    
        # free memory
        del gt_bboxes_per_frame, gt_track_ids_per_frame
        #del mux
    
        return objects

def get_sequence_info(view: fo.DatasetView,
                    gt_field: str,  # fiftyone field name
                    sequence_list: list = None,
                    ): 


    sequence_info = {}
    sequence_names = get_relevant_fields(view, [gt_field, 'sequence']).distinct("sequence")
    if sequence_list is None:
        sequence_list = sequence_names

    for sequence_name in tqdm(sequence_list):
        sequence_view = view.match(F("sequence") == sequence_name)
        sequence_info[sequence_name] = compute_sizes(
            view=sequence_view,
            gt_field=gt_field,
        )
        
    return sequence_info
        
def box_denormalize(boxes: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """
    Denormalizes boxes from [0, 1] to [0, img_w] and [0, img_h].
    Args:
        boxes (Tensor[N, 4]): boxes which will be denormalized.
        img_w (int): Width of image.
        img_h (int): Height of image.

    Returns:
        Tensor[N, 4]: Denormalized boxes.
    """
    if boxes.size == 0:
        return boxes

    # check if boxes are normalized
    if np.any(boxes > 1.0):
        return boxes

    boxes[0::2] *= img_w
    boxes[1::2] *= img_h
    return boxes

def box_convert(boxes: np.ndarray, in_fmt: str, out_fmt: str) -> np.ndarray:
    """
    Converts boxes from given in_fmt to out_fmt.
    Supported in_fmt and out_fmt are:

    'xyxy': boxes are represented via corners, x1, y1 being top left and x2, y2 being bottom right.
    This is the format that torchvision utilities expect.

    'xywh' : boxes are represented via corner, width and height, x1, y2 being top left, w, h being width and height.

    'cxcywh' : boxes are represented via centre, width and height, cx, cy being center of box, w, h
    being width and height.

    Args:
        boxes (Tensor[N, 4]): boxes which will be converted.
        in_fmt (str): Input format of given boxes. Supported formats are ['xyxy', 'xywh', 'cxcywh'].
        out_fmt (str): Output format of given boxes. Supported formats are ['xyxy', 'xywh', 'cxcywh']

    Returns:
        Tensor[N, 4]: Boxes into converted format.
    """
    if boxes.size == 0:
        return boxes

    allowed_fmts = ("xyxy", "xywh", "cxcywh")
    if in_fmt not in allowed_fmts or out_fmt not in allowed_fmts:
        raise ValueError(
            "Unsupported Bounding Box Conversions for given in_fmt and out_fmt")

    if in_fmt == out_fmt:
        return boxes.copy()

    if in_fmt != "xyxy" and out_fmt != "xyxy":
        # convert to xyxy and change in_fmt xyxy
        if in_fmt == "xywh":
            boxes = _box_xywh_to_xyxy(boxes)
        elif in_fmt == "cxcywh":
            boxes = _box_cxcywh_to_xyxy(boxes)
        in_fmt = "xyxy"

    if in_fmt == "xyxy":
        if out_fmt == "xywh":
            boxes = _box_xyxy_to_xywh(boxes)
        elif out_fmt == "cxcywh":
            boxes = _box_xyxy_to_cxcywh(boxes)
    elif out_fmt == "xyxy":
        if in_fmt == "xywh":
            boxes = _box_xywh_to_xyxy(boxes)
        elif in_fmt == "cxcywh":
            boxes = _box_cxcywh_to_xyxy(boxes)
    return boxes

def _box_xywh_to_xyxy(boxes):
    """
    Converts bounding boxes from (x, y, w, h) format to (x1, y1, x2, y2) format.
    (x, y) refers to top left of bounding box.
    (w, h) refers to width and height of box.
    Args:
        boxes (ndarray[N, 4]): boxes in (x, y, w, h) which will be converted.

    Returns:
        boxes (ndarray[N, 4]): boxes in (x1, y1, x2, y2) format.
    """
    x, y, w, h = np.split(boxes, 4, axis=-1)
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    converted_boxes = np.concatenate([x1, y1, x2, y2], axis=-1)
    return converted_boxes

def _box_cxcywh_to_xyxy(boxes):
    """
    Converts bounding boxes from (cx, cy, w, h) format to (x1, y1, x2, y2) format.
    (cx, cy) refers to center of bounding box
    (w, h) are width and height of bounding box
    Args:
        boxes (ndarray[N, 4]): boxes in (cx, cy, w, h) format which will be converted.

    Returns:
        boxes (ndarray[N, 4]): boxes in (x1, y1, x2, y2) format.
    """
    cx, cy, w, h = np.split(boxes, 4, axis=-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    converted_boxes = np.concatenate([x1, y1, x2, y2], axis=-1)
    return converted_boxes

def _box_xyxy_to_xywh(boxes):
    """
    Converts bounding boxes from (x1, y1, x2, y2) format to (x, y, w, h) format.
    (x1, y1) refer to top left of bounding box
    (x2, y2) refer to bottom right of bounding box
    Args:
        boxes (ndarray[N, 4]): boxes in (x1, y1, x2, y2) which will be converted.

    Returns:
        boxes (ndarray[N, 4]): boxes in (x, y, w, h) format.
    """
    x1, y1, x2, y2 = np.split(boxes, 4, axis=-1)
    w = x2 - x1
    h = y2 - y1
    converted_boxes = np.concatenate([x1, y1, w, h], axis=-1)
    return converted_boxes

def _box_xyxy_to_cxcywh(boxes):
    """
    Converts bounding boxes from (x1, y1, x2, y2) format to (cx, cy, w, h) format.
    (x1, y1) refer to top left of bounding box
    (x2, y2) refer to bottom right of bounding box
    Args:
        boxes (ndarray[N, 4]): boxes in (x1, y1, x2, y2) format which will be converted.

    Returns:
        boxes (ndarray[N, 4]): boxes in (cx, cy, w, h) format.
    """
    x1, y1, x2, y2 = np.split(boxes, 4, axis=-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    converted_boxes = np.concatenate([cx, cy, w, h], axis=-1)
    return converted_boxes

def results_to_df(metrics, sequence_list: list = None) -> pd.DataFrame:
    # if sequence_list is not provided, compute for all sequences
    df = pd.DataFrame()
    if sequence_list is None:
        sequence_list = metrics.accumulators.keys()
    for sequence in sequence_list:
        summary = metrics.compute(sequence=sequence)
        row = pd.DataFrame(summary)
        row['sequence'] = sequence
        df = pd.concat([df, row])
    # df.set_index('sequence', inplace=True)
    df['mota'] = df['mota']*100
    df['motp'] = (1-df['motp'])*100
    return df

def classify_num_objects(x):
    n_objects_ranges_tuples = [
        ("zero", [0, 1]), 
        ("one", [1, 2]), 
        ("two", [2, 3]),
        ("few", [3,7]),
        ("many", [7, 20])
    ]
    category = None
    for label, n_objects_range in n_objects_ranges_tuples:
            if n_objects_range[0] <= x < n_objects_range[1]:
                category = label
                break
    return category