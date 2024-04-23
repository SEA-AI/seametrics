
import numpy as np
import fiftyone as fo
import torch
from tqdm import tqdm
from PIL import Image
from transformers import MaskFormerForInstanceSegmentation, MaskFormerImageProcessor
from cleanlab.segmentation.rank import get_label_quality_scores, issues_from_scores 
import cv2
import albumentations as A
import torch.nn.functional as NNF

ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

image_transform = A.Compose([
    A.Resize(width=512, height=512),
    A.Normalize(mean=ADE_MEAN, std=ADE_STD),
])

def transform_mask_to_image_size(image_size, bbox, mask, index):
    height, width = image_size
    image_size_mask = np.zeros(image_size, dtype=np.uint8)
    
    # Unpack bounding box coordinates
    left, top, bbox_width, bbox_height = bbox
    # Calculate absolute coordinates
    abs_top = int(top * height)
    abs_left = int(left * width)
    abs_bottom = min(abs_top + int(bbox_height * height), height)
    abs_right = min(abs_left + int(bbox_width * width), width)
    
    # Calculate the actual height and width of the mask area
    actual_height = abs_bottom - abs_top
    actual_width = abs_right - abs_left

    # Resize the mask if its dimensions do not match the bounding box slice dimensions
    if (mask.shape[0] != actual_height) or (mask.shape[1] != actual_width):
        # If resizing is needed, consider using interpolation methods appropriate for mask data
        resized_mask = np.zeros((actual_height, actual_width), dtype=mask.dtype)
        scale_y = actual_height / mask.shape[0]
        scale_x = actual_width / mask.shape[1]
        for i in range(actual_height):
            for j in range(actual_width):
                orig_y = int(i / scale_y)
                orig_x = int(j / scale_x)
                resized_mask[i, j] = mask[orig_y, orig_x]
        mask = resized_mask
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
    for annotation in annotations:
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


def compute_and_upload_softmin(dataset_view, model_path: str, target_size, mask_field_to_save: str, softmin_score_field: str, device=None):
    pred_probs_list = []
    ground_truth_labels_list = []

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MaskFormerForInstanceSegmentation.from_pretrained(model_path).to(device)
    processor = MaskFormerImageProcessor.from_pretrained(model_path, use_tensors=True)

    label_to_id = model.config.label2id
    model_labels =  model.config.id2label
    class_names = [v for k,v in model_labels.items()]

    all_softmin_scores = []
    
    for sample in tqdm(dataset_view):
        image_filepath = sample.filepath
        image = Image.open(image_filepath).convert('RGB')

        pixel_values = image_transform(image=np.array(image))["image"]
        pixel_values = np.moveaxis(pixel_values, -1, 0)
        pixel_values = torch.from_numpy(pixel_values).unsqueeze(0)

        with torch.no_grad():
            outputs = model(pixel_values.to(device))

        class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
        masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]

        # Compute probabilities and remove the null class `[..., :-1]`
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

        # Compute semantic segmentation probabilities of shape (batch_size, num_classes, height, width)
        pred_probs = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
        resized_probs = NNF.interpolate(
                pred_probs, size=target_size, mode='bilinear', align_corners=False
            )
        pred_probs_np = resized_probs.cpu().numpy()


        ground_truth_label = create_image_mask(sample['ground_truth_det.detections'], (image.size[1], image.size[0]), label_to_id)

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