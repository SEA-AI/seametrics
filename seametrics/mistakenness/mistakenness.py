import os
from pathlib import Path

# Set configuration path for FiftyOne
config_path = Path.home() / ".fiftyone" / "config.global_mongodb.json"
os.environ["FIFTYONE_CONFIG_PATH"] = str(config_path)

import fiftyone as fo
import fiftyone.brain as fob
import yolov5
import cv2
import yaml 

# Function to load class map from YAML file
def load_class_map(class_map_file_path):
    with open(class_map_file_path, 'r') as file:
        return yaml.safe_load(file)

# Function to cast label based on class map
def cast_label(label, cast_dict):
    if label in cast_dict:
        return cast_dict[label]
    return label

# Function to cast labels in a dataset using a class map
def cast_labels(dataset_name: str, sample_field: str, class_map_file_path: str):
    class_map = load_class_map(class_map_file_path)
    dataset = fo.load_dataset(dataset_name)

    for sample in dataset.iter_samples(progress=True):
        detections_list = []
        if sample.ground_truth_det is not None:
            for det in sample.ground_truth_det.detections:
                label = cast_label(det.label, class_map)
                if label == "None":
                    continue
                
                detection = fo.Detection(label=label, bounding_box=det.bounding_box, index=det.index)
                detections_list.append(detection)
        
        det_list = fo.core.labels.Detections(detections=detections_list)
        sample[sample_field] = det_list
        sample.save()
    dataset.save()

# Function to add YOLOv5 predictions to a dataset
def add_yolov5_predictions(dataset_name: str, sample_field: str):
    # Load YOLOv5 model
    model = yolov5.load('SEA-AI/yolov5n-IR')

    # Set model parameters
    model.conf = 0.1  # NMS confidence threshold
    model.iou = 0.1  # NMS IoU threshold
    model.agnostic = True  # NMS class-agnostic

    # Load dataset
    dataset = fo.load_dataset(dataset_name)

    # Iterate over samples in the dataset
    for sample in dataset.iter_samples(progress=True):
        if "ADDED_YOLO_PREDICTION" in sample.tags:
            continue

        # Read frame from sample
        frame = cv2.imread(sample.filepath)
        h, w, _ = frame.shape

        # Perform inference using YOLOv5
        results = model(frame, size=640)

        detections = []
        for *box, conf, cls in results.xyxy[0]:
            # FiftyOne expects [top-left-x, top-left-y, width, height] normalized to [0, 1]
            x1, y1, x2, y2 = box
            rel_box = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]
            detections.append(
                fo.Detection(
                    label=model.names[int(cls)],
                    bounding_box=rel_box,
                    confidence=conf,
                )
            )

        det_list = fo.core.labels.Detections(detections=detections)
        sample[sample_field] = det_list
        sample.tags.append("ADDED_YOLO_PREDICTION")
        sample.save()
        
    dataset.save()

# Function to add mistakenness localization metric to a dataset
def add_mistakenness_loc_metric(dataset, label_field="ground_truth_det"):
    # Specify field as float to prevent cast issues
    dataset.add_sample_field("mistakenness_loc", fo.FloatField)

    # Loop through samples
    for sample in dataset:
        detections = sample[label_field]

        # Extract the mistakenness_loc values from detections
        mistakenness_loc_values = [
            det.mistakenness_loc for det in detections.detections if hasattr(det, "mistakenness_loc")
        ]

        # Assign the maximum mistakenness_loc to the sample field, or -1.0 if there are no detections
        sample["mistakenness_loc"] = max(mistakenness_loc_values, default=-1.0)
        sample.save()
    dataset.save()

# Function to compute mistakenness metric for a dataset
def compute_mistakenness(dataset_name, brain_key="mistakenness", pred_field="yolo_prediction", label_field="ground_truth_det_yolo"):
    dataset = fo.load_dataset(dataset_name)
    # Add this field to prevent cast issued when calculating mistakenness (otherwise the mistakeness score will possibly be casted to integers)
    dataset.add_sample_field(brain_key, fo.FloatField)

    """
        To customize the default settings for mistakenness computation, update the values in the mistakenness.py file located in the fiftyone.brain module:
        (fiftyone/brain/internal/core/mistakenness.py lines 34-35)

        _MISSED_CONFIDENCE_THRESHOLD = 0.5  # Threshold for detection confidence to be considered as missed
        _DETECTION_IOU = 0.1  # Threshold for Intersection over Union (IoU) to consider a detection as a match
    """

    fob.compute_mistakenness(dataset, pred_field=pred_field, label_field=label_field)
    add_mistakenness_loc_metric(dataset, label_field="ground_truth_det_yolo")


# Main function
if __name__ == "__main__":
    # Define dataset name
    print("Loading dataset...")
    dataset_name = "TRAIN_RL_SPLIT_THERMAL_2024_03"
    
    # Cast labels
    print("Casting labels...")
    cast_labels(dataset_name=dataset_name, sample_field="ground_truth_det_yolo", class_map_file_path="class_map.yaml")

    # Add YOLOv5 predictions to the dataset
    print("Inference using YOLOv5...")
    add_yolov5_predictions(dataset_name=dataset_name, sample_field="yolo_prediction")

    # Add yolo predictions to the dataset
    print("Computing mistakenness...")
    compute_mistakenness(dataset_name=dataset_name)
