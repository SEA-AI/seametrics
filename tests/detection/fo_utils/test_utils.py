
import evaluate

from seametrics.fo_utils.utils import _add_batch
from seametrics.fo_utils.utils import _fo_dets_to_metrics_dict

from dependencies import main_code_part
from dependencies import main_code_part_empty
from dependencies import extract_payload



def test_det_metrics():

    # Example 1: True Positive example on the large range
    ######################################
    gt_bbox = [100.0, 100.0, 300.0, 300.0]
    pred_bbox = [100.0, 100.0, 300.0, 300.0]
    payload = main_code_part(gt_bbox, pred_bbox)

    gt_normalized, pred_normalized, img_res = extract_payload(payload)

    processed_pred = _fo_dets_to_metrics_dict(
        fo_dets=pred_normalized, w=img_res[1], h=img_res[0]
    )
    processed_gt = _fo_dets_to_metrics_dict(
        fo_dets=gt_normalized,
        w=img_res[1],
        h=img_res[0],
        is_ground_truth=True,
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



def test_det_metrics_add_batch():

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
    img_res = (512, 640)

    module = evaluate.load("SEA-AI/det-metrics", area_ranges_tuples=area_ranges_tuples)

    module_output, predictions, references = _add_batch(module, payload, None)
    bounding_box_pred = payload["sequences"][
        "Sentry_2023_05_France_FB_WL_2023_05_16_15_23_57"
    ]["volcanic-sweep-3_02_2023_N_LN1_ep288_CNN"][0][0]["bounding_box"]
    bounding_box_gt = payload["sequences"][
        "Sentry_2023_05_France_FB_WL_2023_05_16_15_23_57"
    ]["annotations_sf"][0][0]["bounding_box"]

    assert module is module_output
    assert len(predictions[0][0]) == 4
    assert len(references[0][0]) == 4

    assert int(predictions[0][0][0]) == int(bounding_box_pred[0] * img_res[1])
    assert int(predictions[0][0][1]) == int(bounding_box_pred[1] * img_res[0])
    assert int(predictions[0][0][2]) == int(bounding_box_pred[2] * img_res[1])
    assert int(predictions[0][0][3]) == int(bounding_box_pred[3] * img_res[0])

    assert int(references[0][0][0]) == int(bounding_box_gt[0] * img_res[1])
    assert int(references[0][0][1]) == int(bounding_box_gt[1] * img_res[0])
    assert int(references[0][0][2]) == int(bounding_box_gt[2] * img_res[1])
    assert int(references[0][0][3]) == int(bounding_box_gt[3] * img_res[0])

    assert module is not None
    assert module_output is not None

    assert predictions == [[[125.0, 80.0, 375.0, 240.0]]]
    assert references == [[[125.0, 80.0, 375.0, 240.0]]]

    payload = main_code_part_empty(gt_bbox, pred_bbox)
    img_res = (512, 640)

    module = evaluate.load("SEA-AI/det-metrics", area_ranges_tuples=area_ranges_tuples)

    module_output, predictions, references = _add_batch(module, payload, None)
    bounding_box_pred = payload["sequences"][
        "Sentry_2023_05_France_FB_WL_2023_05_16_15_23_57"
    ]["volcanic-sweep-3_02_2023_N_LN1_ep288_CNN"]
    bounding_box_gt = payload["sequences"][
        "Sentry_2023_05_France_FB_WL_2023_05_16_15_23_57"
    ]["annotations_sf"]

    assert references == [[]]
    assert predictions == [[]]

    assert module is not None
    assert module_output is not None

    assert bounding_box_pred == [[]]
    assert bounding_box_gt == [[]]

    assert len(predictions[0]) == 0
    assert len(references[0]) == 0

    assert module is not None
    assert module_output is not None

    assert predictions == [[]]
    assert references == [[]]