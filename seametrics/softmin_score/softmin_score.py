import torch
from tqdm import tqdm
from transformers import MaskFormerForInstanceSegmentation, MaskFormerImageProcessor
import cv2
import numpy as np
import torch.nn.functional as NNF
from fiftyone import ViewField as F
import fiftyone as fo
from cleanlab.segmentation.rank import get_label_quality_scores, issues_from_scores 
from cleanlab.segmentation.filter import find_label_issues 
import time
from PIL import Image
import albumentations as A

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_NAME = "TRAIN_PANOPTIC_DATASET"
MODEL_PATH =  "/mnt/hdd/maryam/models/model_panoptik_27k_saved_100epochs/last_model"
MODEL_NAME = "maskformer-27k-100epochs"
FIELD_TO_SAVE = "softmin_errors"
NUM_SAMPLES = 10


ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

image_transform = A.Compose([
    A.Resize(width=512, height=512),
    A.Normalize(mean=ADE_MEAN, std=ADE_STD),
])

dataset = fo.load_dataset(DATASET_NAME) 

# dataset_view = dataset.match(F("filepath")=="/mnt/fiftyoneDB/Database/Image_Data/Thermal_Images_8Bit/Trip_46_Seq_7/1355061_l.jpg") #change this to be smaller than the entire dataset if you want to do a quick test
# dataset_view = dataset.select_group_slices()
# dataset_view = dataset.select_group_slices().take(NUM_SAMPLES)
dataset_view = dataset.take(NUM_SAMPLES)


model = MaskFormerForInstanceSegmentation.from_pretrained(MODEL_PATH).to(DEVICE)
processor = MaskFormerImageProcessor.from_pretrained(MODEL_PATH, use_tensors=True)

label_to_id = model.config.label2id
model_labels =  model.config.id2label
class_names = [v for k,v in model_labels.items()]

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
    
    # Place mask in image mask
    image_size_mask[abs_top:abs_bottom, abs_left:abs_right] = mask * index

    return image_size_mask

def create_image_mask(annotations, image_size):
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


all_softmin_scores = []

time0 = time.time()
for sample in tqdm(dataset_view):
    image_filepath = sample.filepath
    # image = Image.open(image_filepath).convert('RGB')
    image = np.array(Image.open(image_filepath))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    pixel_values = image_transform(image=np.array(image))["image"]
    pixel_values = np.moveaxis(pixel_values, -1, 0)
    pixel_values = torch.from_numpy(pixel_values).unsqueeze(0)

    with torch.no_grad():
        outputs = model(pixel_values.to(DEVICE))


    class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
    masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]

    # Compute probabilities and remove the null class `[..., :-1]`
    masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
    masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

    # Compute semantic segmentation probabilities of shape (batch_size, num_classes, height, width)
    pred_probs = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
    target_sizes=[image.size[::-1]]
    for idx in range(len(pred_probs)):
        resized_probs = NNF.interpolate(
            pred_probs[idx].unsqueeze(dim=0), size=target_sizes[idx], mode='bilinear', align_corners=False
        )
    pred_probs_np = resized_probs.cpu().numpy()

    # ground_truth_label = create_image_mask(sample['ground_truth_det.detections'], image.size)
    image_size = (image.size[1], image.size[0])

    ground_truth_label = np.expand_dims(create_image_mask(sample['ground_truth_det.detections'], image_size), axis=0)

    issues = find_label_issues(ground_truth_label, pred_probs_np, downsample = 4, n_jobs=None, batch_size=1)
    image_scores, pixel_scores = get_label_quality_scores(labels=ground_truth_label, pred_probs=pred_probs_np)
    issues_from_score = issues_from_scores(image_scores, pixel_scores, threshold=0.5)

    issues_mask = np.squeeze(issues_from_score.astype(int))

    fo_annotations = softmin_output_to_fo_format(issues_mask)

    # Softmin_score ranges from 0 to 1, such that lower scores indicate images more likely to contain some mislabeled pixels.
    sample['softmin_score'] = round(image_scores[0],4)
    
    all_softmin_scores.append(sample['softmin_score'])

    #save in the required field in fiftyone sample
    sample[FIELD_TO_SAVE] = fo.Detections(detections=fo_annotations)
    sample.save()

time1 = time.time()
print(f"It took {str(time1-time0)} for {NUM_SAMPLES}")

avg_softmin_score = np.mean(all_softmin_scores)

print("Overall average softmin score: ", avg_softmin_score)