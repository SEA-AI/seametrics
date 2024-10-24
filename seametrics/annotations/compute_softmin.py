
import numpy as np
from fiftyone import ViewField as F
import fiftyone as fo
from utils import compute_and_upload_softmin
import time
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_NAME = "SAILING_PANOPTIC_DATASET_QA"
MODEL_PATH = "SEA-AI/maskformer-27k-100epochs"
MODEL_NAME = "maskformer-27k-100epochs"
SOFTMIN_MASK_FIELD = "softmin_errors"
SOFTMIN_SCORE_FIELD = "softmin_score"
PRED_PROBS_FIELD = "ps_pred_probs"
NUM_SAMPLES = 1000
TARGET_SIZE = (640, 512)

dataset = fo.load_dataset(DATASET_NAME) 

# dataset_view = dataset.match(F("filepath")=="/mnt/fiftyoneDB/Database/Image_Data/Thermal_Images_8Bit/Trip_46_Seq_7/1355061_l.jpg") #change this to be smaller than the entire dataset if you want to do a quick test
# dataset_view = dataset.select_group_slices()
# dataset_view = dataset.select_group_slices().take(NUM_SAMPLES)
if NUM_SAMPLES:
    dataset = dataset.take(NUM_SAMPLES)

print(f"Calculating softmin score on {str(len(dataset))} samples")

time0 = time.time()
avg_softmin_score, pred_probs_all_np, ground_truth_labels_all_np = compute_and_upload_softmin(dataset, 
                                                                                              MODEL_PATH, 
                                                                                              TARGET_SIZE, 
                                                                                              PRED_PROBS_FIELD, 
                                                                                              SOFTMIN_MASK_FIELD, 
                                                                                              SOFTMIN_SCORE_FIELD, 
                                                                                              device=DEVICE)

time1 = time.time()

print(f"It took {str(time1-time0)} for {str(NUM_SAMPLES)}")

print("Overall average softmin score: ", avg_softmin_score)