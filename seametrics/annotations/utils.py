import os
import cv2
import numpy as np
import fiftyone as fo
import torch
import json
from tqdm import tqdm
import albumentations as A
from cleanlab.segmentation.rank import get_label_quality_scores, issues_from_scores 


ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

image_transform = A.Compose([
    A.Resize(width=512, height=512),
    A.Normalize(mean=ADE_MEAN, std=ADE_STD),
])

def get_label2id(model_path, device=None):
    if os.path.exists("label2id.json"):
        with open("label2id.json", "r") as json_file:
            label2id = json.load(json_file)
    else:
        from transformers import MaskFormerForInstanceSegmentation
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MaskFormerForInstanceSegmentation.from_pretrained(model_path).to(device)
        label2id = model.config.label2id
    return label2id



def transform_mask_to_image_size(image_size, bbox, mask, index):
    height, width = image_size
    image_size_mask = np.zeros(image_size, dtype=np.uint8)
    
    # Unpack bounding box coordinates
    left, top, bbox_width, bbox_height = bbox
    
    # Calculate absolute coordinates
    abs_top = round(top * height)
    abs_left = round(left * width)
    abs_bottom = min(abs_top + round(bbox_height * height), height)
    abs_right = min(abs_left + round(bbox_width * width), width)

    # Place mask in image mask
    image_size_mask[abs_top:abs_bottom, abs_left:abs_right] = mask * index

    return image_size_mask


def create_image_mask(annotations, image_size, label_to_id):
    """
    Create an image mask from a list of mask annotations.
    
    Args:
    - annotations: List of annotations, where each annotation is a dictionary with keys 'bounding_box', 'mask', and 'index'.
    - image_size: Tuple (height, width) representing the size of the image mask to be created.
    
    Returns:
    - image_mask: NumPy array representing the image mask with object indexes.
    """
    height, width = image_size
    image_mask = np.zeros((height, width), dtype=np.uint8)   
    sorted_annotations = sorted(annotations, key=lambda det: det["mask"].sum(), reverse=True) # put large masks below smaller masks
    for annotation in sorted_annotations:
        bbox = annotation['bounding_box']
        mask = annotation['mask']
        index = label_to_id[annotation['label']]
        object_mask_full = transform_mask_to_image_size(image_size, bbox, mask, index)
        image_mask[object_mask_full>0] = index

    return image_mask

def normalize_mask(mask, obj_id):
    # Find pixels belonging to the current object
    y_indices, x_indices = np.where(mask == obj_id) 
    # Calculate the bounding box
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()

    # Normalize the bounding box
    top, left = y_min / mask.shape[0], x_min / mask.shape[1]
    width, height = (x_max - x_min) / mask.shape[1], (y_max - y_min) / mask.shape[0]
    normalized_bb = [left, top, width, height]

    # Create a binary mask for the object
    obj_mask = (mask[y_min:y_max+1, x_min:x_max+1] == obj_id).astype(np.uint8)

    return normalized_bb, obj_mask


def softmin_output_to_fo_format(mask):
    objects = np.unique(mask)
    objects = objects[objects != 0]  # Exclude background

    fo_annotations = []
    for obj_id in objects:

        normalized_bb, obj_mask = normalize_mask(mask, obj_id)

        annotation = fo.Detection(bounding_box=normalized_bb, mask=obj_mask, label='softmin_error')
        fo_annotations.append(annotation)

    return fo_annotations


def compute_and_upload_softmin(dataset_view, model_path: str, target_size, pred_probs_field: str, mask_field_to_save: str, softmin_score_field: str, device=None):
    """
    Processes a dataset to compute and store softmin scores and mask annotations for each sample.

    This function loads a pre-trained MaskFormer model for instance segmentation and processes each sample
    in the dataset. It performs semantic segmentation using the model, computes the resized probabilities for each class,
    and compares these against ground truth labels to compute quality scores. Issues identified based on these scores
    are formatted and saved back into the dataset. Additionally, it calculates the average softmin score for all images.

    Parameters:
        dataset_view (iterable): An iterable view of the dataset where each element is a sample containing
                                 metadata and image file paths.
        model_path (str): Path to the directory where the pre-trained MaskFormer model and associated processor
                          are stored.
        pred_probs_field (str): fiftyone field in the dataset were the prediction probabilities are saved.
        target_size (tuple): A tuple (height, width) specifying the new size to which the prediction probabilities
                             and ground truth labels should be resized.
        mask_field_to_save (str): The field name in the fiftyone dataset where the computed mask annotations should be saved.
        softmin_score_field (str): The field name in the fiftyone dataset where the computed softmin scores should be stored.
        device (torch.device, optional): The device on which the computation should be performed. If None, it will
                                         automatically select GPU if available, otherwise CPU.

    Returns:
        tuple: A tuple containing three elements:
            - avg_softmin_score (float): The average softmin score computed across all samples in the dataset.
            - pred_probs_all_np (numpy.ndarray): A numpy array of all prediction probabilities, stacked from all samples.
            - ground_truth_labels_all_np (numpy.ndarray): A numpy array of all ground truth labels, resized and stacked
                                                          from all samples.
    """

    pred_probs_list = []
    ground_truth_labels_list = []

    label_to_id = get_label2id(model_path, device)

    all_softmin_scores = []
    
    for sample in tqdm(dataset_view):

        if pred_probs_field in sample.field_names and sample[pred_probs_field] is not None:
            pred_probs_np = sample[pred_probs_field] 
            ground_truth_label = create_image_mask(sample['ground_truth_det.detections'], (target_size[0], target_size[1]), label_to_id)

            resized_ground_truth_label = cv2.resize(ground_truth_label, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
            pred_probs_list.append(np.squeeze(pred_probs_np))
            ground_truth_labels_list.append(resized_ground_truth_label)

            ground_truth_label = np.expand_dims(resized_ground_truth_label, axis=0)

            image_scores, pixel_scores = get_label_quality_scores(labels=ground_truth_label, pred_probs=pred_probs_np)
            issues_from_score = issues_from_scores(image_scores, pixel_scores, threshold=0.5)

            issues_mask = np.squeeze(issues_from_score.astype(int))

            fo_annotations = softmin_output_to_fo_format(issues_mask)

            # Softmin_score ranges from 0 to 1, such that lower scores indicate images more likely to contain some mislabeled pixels.
            sample[softmin_score_field] = round(image_scores[0], 4)
            
            all_softmin_scores.append(sample[softmin_score_field])

            #save in the required field in fiftyone sample
            sample[mask_field_to_save] = fo.Detections(detections=fo_annotations)
            sample.save()

    avg_softmin_score = np.mean(all_softmin_scores)
    pred_probs_all_np = np.stack(pred_probs_list, axis=0)
    ground_truth_labels_all_np = np.stack(ground_truth_labels_list, axis=0)

    return avg_softmin_score, pred_probs_all_np, ground_truth_labels_all_np