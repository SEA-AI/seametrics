import os
from pathlib import Path

config_path = Path.home() / ".fiftyone" / "config.global_mongodb.json"
os.environ["FIFTYONE_CONFIG_PATH"] = str(config_path)

import fiftyone as fo
import fiftyone.brain as fob
import yolov5
import cv2
import torch
from ultralytics import YOLO
import yaml


def load_class_map(class_map_file_path: str):
    with open(class_map_file_path, "r") as file:
        return yaml.safe_load(file)


def cast_label(label: str, cast_dict: dict):
    if label in cast_dict:
        return cast_dict[label]
    return label


def cast_labels(
    dataset: fo.core.dataset.Dataset,
    class_map_file_path: str,
    sample_field: str = "ground_truth_det_yolo",
):
    class_map = load_class_map(class_map_file_path)

    for sample in dataset.iter_samples(progress=True):
        detections_list = []

        if sample.ground_truth_det is not None:
            for det in sample.ground_truth_det.detections:
                label = cast_label(det.label, class_map)

                # if label is None than it is igored in training and should also be skipped here
                if label == "None":
                    continue

                detection = fo.Detection(label=label,
                                         bounding_box=det.bounding_box,
                                         index=det.index)
                detections_list.append(detection)

        det_list = fo.core.labels.Detections(detections=detections_list)
        sample[sample_field] = det_list

        sample.save()

    dataset.save()


def add_region_predicitions(
    dataset: fo.core.dataset.Dataset,
    weights_file_path: str,
    sample_field_ground_truth: str = "region_ground_truth",
    sample_field_prediction: str = "region_prediction",
):
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(weights_file_path).to(device)

    classes = ["LAKE", "HARBOUR", "OPEN_SEA", "SHORELINE"]

    for sample in dataset.iter_samples(progress=True):
        # Make region prediction
        region_prediction = model.predict(sample.filepath, verbose=False)

        # Extract label and confidence
        label = classes[region_prediction[0].cpu().probs.numpy().top1]
        confidence = region_prediction[0].cpu().probs.numpy().top1conf

        # Add fields to sample
        sample[sample_field_ground_truth] = fo.Classification(
            label=sample.region)
        sample[sample_field_prediction] = fo.Classification(
            label=label, confidence=confidence)

        sample.save()

    dataset.save()


def add_yolov5_predictions(dataset: fo.core.dataset.Dataset,
                           sample_field: str):
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = yolov5.load("SEA-AI/yolov5n-IR").to(device)

    # Configure model parameters
    model.conf = 0.1  # NMS confidence threshold
    model.iou = 0.1  # NMS IoU threshold
    model.agnostic = True  # NMS class-agnostic

    for sample in dataset.iter_samples(progress=True):
        if "ADDED_YOLO_PREDICTION" in sample.tags:
            continue

        # Load image
        frame = cv2.imread(sample.filepath)
        h, w, _ = frame.shape

        # Make prediciton
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
                ))

        det_list = fo.core.labels.Detections(detections=detections)
        sample[sample_field] = det_list
        sample.tags.append("ADDED_YOLO_PREDICTION")

        sample.save()

    dataset.save()


def add_mistakenness_loc_metric(
    dataset: fo.core.dataset.Dataset,
    label_field: str = "ground_truth_det",
    sample_field: str = "mistakenness_loc",
):
    # Specify field as float to prevent cast issues
    dataset.add_sample_field(sample_field, fo.FloatField)

    for sample in dataset.iter_samples(progress=True):
        detections = sample[label_field]

        # Extract the mistakenness_loc values from detections
        mistakenness_loc_values = [
            det.mistakenness_loc for det in detections.detections
            if hasattr(det, "mistakenness_loc")
        ]

        # Assign the maximum mistakenness_loc to the sample field, or -1.0 if there are no detections
        sample["mistakenness_loc"] = max(mistakenness_loc_values, default=-1.0)

        sample.save()

    dataset.save()


def compute_mistakenness(
    dataset: fo.core.dataset.Dataset,
    mistakenness_field_detections: str = "mistakenness",
    mistakenness_field_region_classification: str = "mistakenness_region",
    pred_field: str = "yolo_prediction",
    label_field: str = "ground_truth_det_yolo",
):
    """
    IMPORTANT!!!
    To customize the default settings for mistakenness computation, update the values in the mistakenness.py file located in the fiftyone.brain module:
    (fiftyone/brain/internal/core/mistakenness.py lines 34-35)

    _MISSED_CONFIDENCE_THRESHOLD = 0.5  # Threshold for detection confidence to be considered as missed
    _DETECTION_IOU = 0.01  # Threshold for Intersection over Union (IoU) to consider a detection as a match
    """

    # Add this fields to prevent cast issued when calculating mistakenness (otherwise the mistakeness score will possibly be casted to integers)
    dataset.add_sample_field(mistakenness_field_detections, fo.FloatField)
    dataset.add_sample_field(mistakenness_field_region_classification,
                             fo.FloatField)

    # Compute mistakenness for object detections
    fob.compute_mistakenness(
        dataset,
        pred_field=pred_field,
        label_field=label_field,
        mistakenness_field=mistakenness_field_detections,
    )

    # Add mistaknness_loc field on sample level
    add_mistakenness_loc_metric(dataset, label_field="ground_truth_det_yolo")

    # Compute mistakenness for region classification
    fob.compute_mistakenness(
        dataset,
        pred_fiels="region_prediction",
        label_field="region_ground_truth",
        mistakenness_field=mistakenness_field_region_classification,
    )


def main():
    dataset_name = "TRAIN_RL_SPLIT_THERMAL_2024_03"

    if dataset_name not in fo.list_datasets():
        print("No dataset with specified name found")
        return

    # Loading dataset
    print("Loading dataset...")
    dataset = fo.load_dataset(dataset_name)

    # Cast labels
    print("Casting labels...")
    cast_labels(
        dataset=dataset,
        sample_field="ground_truth_det_yolo",
        class_map_file_path="class_map.yaml",
    )

    # Add yolo predictions to the dataset
    print("Adding yolo predictions...")
    add_yolov5_predictions(dataset=dataset, sample_field="yolo_prediction")

    # Add region prediction to the dataset
    print("Adding region predictions...")
    add_region_predicitions(dataset=dataset,
                            weights_file_path="./weights/best.pt")

    # Computing mistakenness
    print("Computing mistakenness...")
    compute_mistakenness(dataset=dataset)

    print("Successfully computed mistakenness metric!")


if __name__ == "__main__":
    main()
