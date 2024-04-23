import numpy as np
import fiftyone as fo

def payload_to_seg_metric(payload: dict, model_name: str, label2id: dict=None):
    """
    Convert data in the standard payload format to the format expected by the segmentation metric.
     * merges all masks into one image
     * smaller masks are put on top of larger masks

    Args:
        payload (dict): The standard payload dictionary.
        model_name (str): The name of the model field that holds the predictions to be evaluated.
        label2id (dict, optional): The dictionary mapping labels to IDs. Defaults to None.

    Returns:
        tuple: A tuple containing the segmentation metric as a numpy array and the label2id dictionary.
    """

    if label2id is None:
        label2id = dict()
    
    pred_frames = []
    gt_frames = []
    for seq in payload["sequence_list"]:
        h, w = payload["sequences"][seq]["resolution"]
        preds = payload["sequences"][seq][model_name] # n_frames, m_detections
        gts = payload["sequences"][seq][payload["gt_field_name"]] # n_frames, m_detections
        for frame_dets in preds:
            pred_frames.append(multiple_masks_to_single_mask(frame_dets, h, w, label2id))
        for frame_dets in gts:
            gt_frames.append(multiple_masks_to_single_mask(frame_dets, h, w, label2id))

    return np.stack(pred_frames, axis=0), np.stack(gt_frames, axis=0), label2id


def multiple_masks_to_single_mask(frame_dets: list[fo.Detection],
                                  h: int,
                                  w: int,
                                  label2id: dict):
    single_mask = np.zeros((h, w, 2))
    for instance_idx, det in enumerate(sorted(frame_dets, 
                                            key=lambda det: det["mask"].sum(), 
                                            reverse=True)): # put large masks below smaller masks
        start_x, start_y = int(det["bounding_box"][0]*w), int(det["bounding_box"][1]*h)
        y, x = np.where(det["mask"] == 1)
        y += int(start_y)
        x += int(start_x)
        if det["label"] not in label2id:
            label2id[det["label"]] = len(label2id)
        single_mask[y,x] = np.array([label2id[det["label"]], instance_idx])
    return single_mask