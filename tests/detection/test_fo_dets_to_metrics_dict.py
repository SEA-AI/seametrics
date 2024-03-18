import numpy as np
from seametrics.fo_utils.utils import _fo_dets_to_metrics_dict
import uuid


class Detection:
    def __init__(
        self, id_, attributes, tags, label, bounding_box, mask, confidence, index, area
    ):
        self.id = id_
        self.attributes = attributes
        self.tags = tags
        self.label = label
        self.bounding_box = bounding_box
        self.mask = mask
        self.confidence = confidence
        self.index = index
        self.area = area

    @property
    def field_names(self):
        # Get attributes names excluding methods
        return [
            attr
            for attr in self.__dict__
            if not callable(getattr(self, attr)) and not attr.startswith("__")
        ]

    def __getitem__(self, key):
        return getattr(self, key)


def normalize_bbox(gt_bboxes, pred_bboxes):

    for index, item in enumerate(gt_bboxes):
        gt_bboxes[index] = [
            gt_bboxes[index][0] / 512,
            gt_bboxes[index][1] / 640,
            gt_bboxes[index][2] / 512,
            gt_bboxes[index][3] / 640,
        ]

    for index, item in enumerate(pred_bboxes):
        pred_bboxes[index] = [
            pred_bboxes[index][0] / 512,
            pred_bboxes[index][1] / 640,
            pred_bboxes[index][2] / 512,
            pred_bboxes[index][3] / 640,
        ]

    return gt_bboxes, pred_bboxes


def fake_payload(gt_bboxes, pred_bboxes, from_nFoV: bool = False):
    payload = {
        "dataset": "SENTRY_VIDEOS_DATASET_QA",
        "models": ["volcanic-sweep-3_02_2023_N_LN1_ep288_CNN"],
        "gt_field_name": "annotations_sf",
        "sequences": {
            "Sentry_2023_05_France_FB_WL_2023_05_16_15_23_57": {
                "resolution": (512, 640),
                "volcanic-sweep-3_02_2023_N_LN1_ep288_CNN": [[]],
                "annotations_sf": [[]],
            }
        },
        "sequence_list": ["Sentry_2023_05_France_FB_WL_2023_05_16_15_23_57"],
    }

    pred_list = []
    for index, item in enumerate(pred_bboxes):
        var = {
            "id": str(uuid.uuid4()),  # Random UUID
            "attributes": {},
            "tags": [],
            "label": "SHIP",
            "bounding_box": item,
            "mask": None,
            "confidence": 1.0,
            "index": None,
        }

        payload["sequences"]["Sentry_2023_05_France_FB_WL_2023_05_16_15_23_57"][
            "volcanic-sweep-3_02_2023_N_LN1_ep288_CNN"
        ][0].append(var)

    gt_list = []
    for index, item in enumerate(gt_bboxes):

        var = Detection(
            id_=str(uuid.uuid4()),  # Random ID
            attributes={},
            tags=[],
            label="SHIP",
            bounding_box=item,
            mask=None,
            confidence=1.0,
            index=None,
            area=(
                (item[2] * 640) * (item[3] * 512)
                if from_nFoV == False
                else (item[2] * 640) * (item[3] * 512) * 100
            ),
        )

        payload["sequences"]["Sentry_2023_05_France_FB_WL_2023_05_16_15_23_57"][
            "annotations_sf"
        ][0].append(var)

    return payload


def main_code_part(gt_bbox, pred_bbox, nFoV: bool = False):
    gt_bbox1, pred_bbox1 = normalize_bbox([gt_bbox], [pred_bbox])
    payload = fake_payload(gt_bbox1, pred_bbox1, nFoV)
    return payload


def extract_payload(payload):

    model = payload["models"][0]
    sequence = payload["sequence_list"][0]
    seq_data = payload["sequences"][sequence]
    gt_normalized = seq_data[payload["gt_field_name"]][0]  # shape: (n_frames, m_gts)
    pred_normalized = seq_data[model][0]  # shape: (n_frames, l_preds)
    img_res = seq_data["resolution"]  # (h, w)

    return gt_normalized, pred_normalized, img_res


def test_det_metrics():

    area_ranges_tuples = [
        ("all", [0, 1e5**2]),
        ("small", [0, 36]),
        ("medium", [36, 144]),
        ("large", [144, 1e5**2]),
    ]

    # Example 1: True Positive example on the large range
    ######################################
    gt_bbox = [100.0, 100.0, 300.0, 300.0]
    pred_bbox = [100.0, 100.0, 300.0, 300.0]
    payload = main_code_part(gt_bbox, pred_bbox)

    gt_normalized, pred_normalized, img_res = extract_payload(payload)

    processed_pred = _fo_dets_to_metrics_dict(
        fo_dets=pred_normalized, w=img_res[1], h=img_res[0], include_scores=True
    )
    processed_gt = _fo_dets_to_metrics_dict(
        fo_dets=gt_normalized,
        w=img_res[1],
        h=img_res[0],
        include_scores=False,
        include_areas=True,
        gt_data=True,
    )

    assert len(processed_pred[0]["boxes"][0]) == 4
    assert len(processed_pred[0]["scores"]) == 1
    assert len(processed_pred[0]["labels"]) == 1

    assert int(processed_pred[0]["boxes"][0][0]) == int(
        pred_normalized[0]["bounding_box"][0] * img_res[1]
    )
    assert int(processed_pred[0]["boxes"][0][1]) == int(
        pred_normalized[0]["bounding_box"][1] * img_res[0]
    )
    assert int(processed_pred[0]["boxes"][0][2]) == int(
        pred_normalized[0]["bounding_box"][2] * img_res[1]
    )
    assert int(processed_pred[0]["boxes"][0][3]) == int(
        pred_normalized[0]["bounding_box"][3] * img_res[0]
    )

    assert len(processed_gt[0]["boxes"][0]) == 4
    assert len(processed_gt[0]["labels"]) == 1

    assert int(processed_gt[0]["boxes"][0][0]) == int(
        gt_normalized[0]["bounding_box"][0] * img_res[1]
    )
    assert int(processed_gt[0]["boxes"][0][1]) == int(
        gt_normalized[0]["bounding_box"][1] * img_res[0]
    )
    assert int(processed_gt[0]["boxes"][0][2]) == int(
        gt_normalized[0]["bounding_box"][2] * img_res[1]
    )
    assert int(processed_gt[0]["boxes"][0][3]) == int(
        gt_normalized[0]["bounding_box"][3] * img_res[0]
    )

    assert int(processed_gt[0]["area"][0]) == int(
        processed_gt[0]["boxes"][0][2] * processed_gt[0]["boxes"][0][3]
    )
