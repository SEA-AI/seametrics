import numpy as np
import pytest
import torch
from consts import unit_tests_results
from test_panoptic_utils import generate_synthetic_payload

from seametrics.panoptic.pq import PanopticQuality
from seametrics.panoptic.utils import payload_to_seg_metric


@pytest.fixture
def setup_panoptic_quality():
    label2id = {
        "WATER": 0,
        "SKY": 1,
        "LAND": 2,
        "MOTORBOAT": 3,
        "FAR_AWAY_OBJECT": 4,
        "SAILING_BOAT_WITH_CLOSED_SAILS": 5,
        "SHIP": 6,
        "WATERCRAFT": 7,
        "SPHERICAL_BUOY": 8,
        "CONSTRUCTION": 9,
        "FLOTSAM": 10,
        "SAILING_BOAT_WITH_OPEN_SAILS": 11,
        "CONTAINER": 12,
        "PILLAR_BUOY": 13,
        "AERIAL_ANIMAL": 14,
        "HUMAN_IN_WATER": 15,
        "OWN_BOAT": 16,
        "WOODEN_LOG": 17,
        "MARITIME_ANIMAL": 18,
    }
    stuff = ["WATER", "SKY", "LAND", "CONSTRUCTION", "ICE", "OWN_BOAT"]
    per_class = True
    split_sq_rq = True
    area_rng = [(0, 1e5**2), (0**2, 6**2), (6**2, 12**2), (12**2, 1e5**2)]
    pq = PanopticQuality(
        things=set(
            [label2id[label] for label in label2id.keys() if label not in stuff]
        ),
        stuffs=set([label2id[label] for label in label2id.keys() if label in stuff]),
        return_per_class=per_class,
        return_sq_and_rq=split_sq_rq,
        areas=area_rng,
    )
    return pq, label2id


def test_panoptic_quality_initialization(setup_panoptic_quality):
    pq, _ = setup_panoptic_quality
    assert pq.device is not None, "Device should not be None"
    assert pq.metric.return_per_class is True
    assert pq.metric.return_sq_and_rq is True
    assert len(pq.things) == 14
    assert len(pq.stuffs) == 5
    assert len(pq.things) + len(pq.stuffs) == 19
    assert pq.CHUNK_SIZE == 200


def test_panoptic_quality_areas(setup_panoptic_quality):
    pq, _ = setup_panoptic_quality
    assert pq.get_areas() == [
        (0, 1e5**2),
        (0**2, 6**2),
        (6**2, 12**2),
        (12**2, 1e5**2),
    ], "Areas are not set correctly"


def test_update_with_numpy_array(setup_panoptic_quality):
    pq, _ = setup_panoptic_quality

    preds = np.array([[[[3, 1], [4, 1]], [[7, 0], [8, 0]]]])
    targets = np.array([[[[3, 1], [4, 1]], [[7, 0], [8, 0]]]])

    pq.update(preds, targets)
    assert pq.metric.true_positives.sum() > 0, "True positives should be updated"


def test_update_with_torch_tensor(setup_panoptic_quality):
    pq, _ = setup_panoptic_quality

    preds = torch.tensor([[[[3, 1], [4, 1]], [[7, 0], [8, 0]]]])
    targets = torch.tensor([[[[3, 1], [4, 1]], [[7, 0], [8, 0]]]])

    pq.update(preds, targets)
    assert pq.metric.true_positives.sum() > 0, "True positives should be updated"


def test_compute_metric(setup_panoptic_quality):
    pq, label2id = setup_panoptic_quality

    payload = generate_synthetic_payload(gt_config=[1, 2], model_config=[1, 0])
    pred, gt, label2id = payload_to_seg_metric(
        payload=payload, model_name="model", label2id=label2id
    )
    pred[pred == -1] = 0
    gt[gt == -1] = 0

    pq.update(pred, gt)
    assert (
        pq.metric.true_positives.sum() == 3 * 2
    )  # The * 2 is because we are counting each
    assert (
        pq.metric.false_positives.sum() == 0 * 2
    )  # object twice due to the area ranges
    assert pq.metric.false_negatives.sum() == 2 * 2

    pq_value, rq_value, sq_value = pq.compute()

    expected_pq = unit_tests_results["test_panoptic_pq"]["test_compute_metric"][
        "pq_value"
    ]
    expected_rq = unit_tests_results["test_panoptic_pq"]["test_compute_metric"][
        "rq_value"
    ]
    expected_sq = unit_tests_results["test_panoptic_pq"]["test_compute_metric"][
        "sq_value"
    ]

    assert torch.allclose(pq_value, expected_pq, atol=1e-4)
    assert torch.allclose(rq_value, expected_rq, atol=1e-4)
    assert torch.allclose(sq_value, expected_sq, atol=1e-4)


def test_update_and_compute(setup_panoptic_quality):
    pq, label2id = setup_panoptic_quality
    pq_1, _ = setup_panoptic_quality
    payload = generate_synthetic_payload(gt_config=[1, 2], model_config=[1, 0])
    pred, gt, label2id = payload_to_seg_metric(
        payload=payload, model_name="model", label2id=label2id
    )
    pred[pred == -1] = 0
    gt[gt == -1] = 0
    pq.update(pred, gt)
    pq_value, rq_value, sq_value = pq.compute()
    pq_1_value, rq_1_value, sq_1_value = pq_1.update_and_compute(pred, gt)

    assert torch.allclose(pq_value, pq_1_value, atol=1e-4)
    assert torch.allclose(rq_value, rq_1_value, atol=1e-4)
    assert torch.allclose(sq_value, sq_1_value, atol=1e-4)


def test_multiple_updates_before_compute(setup_panoptic_quality):
    pq, label2id = setup_panoptic_quality
    pq1, label2id_1 = setup_panoptic_quality

    payload = generate_synthetic_payload(gt_config=[1], model_config=[1])
    pred, gt, label2id = payload_to_seg_metric(
        payload=payload, model_name="model", label2id=label2id
    )
    pq.update(pred, gt)
    payload = generate_synthetic_payload(gt_config=[2], model_config=[1])
    pred, gt, label2id = payload_to_seg_metric(
        payload=payload, model_name="model", label2id=label2id
    )
    pq.update(pred, gt)
    payload = generate_synthetic_payload(gt_config=[0], model_config=[0])
    pred, gt, label2id = payload_to_seg_metric(
        payload=payload, model_name="model", label2id=label2id
    )
    pq.update(pred, gt)
    pq_value, rq_value, sq_value = pq.compute()

    payload = generate_synthetic_payload(gt_config=[1, 2, 0], model_config=[1, 1, 0])
    pred, gt, label2id_1 = payload_to_seg_metric(
        payload=payload, model_name="model", label2id=label2id_1
    )
    pq1.update(pred, gt)
    pq_1_value, rq_1_value, sq_1_value = pq1.compute()

    assert torch.allclose(pq_value, pq_1_value, atol=1e-4)
    assert torch.allclose(rq_value, rq_1_value, atol=1e-4)
    assert torch.allclose(sq_value, sq_1_value, atol=1e-4)


"""
And then, it would also be nice to have one update test, where the areas are split up in two 
 area ranges and there are two ground truths which are not in the same area ranges.
 """
