import numpy as np
import pytest
import torch

from seametrics.panoptic.pq import PanopticQuality

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
    return pq


def test_panoptic_quality_initialization(setup_panoptic_quality):
    pq = setup_panoptic_quality
    assert pq.device is not None, "Device should not be None"
    assert pq.metric.return_per_class is True
    assert pq.metric.return_sq_and_rq is True
    assert len(pq.things) == 14
    assert len(pq.stuffs) == 5
    assert len(pq.things) + len(pq.stuffs) == 19
    assert pq.CHUNK_SIZE == 200


def test_panoptic_quality_areas(setup_panoptic_quality):
    pq = setup_panoptic_quality
    assert pq.get_areas() == [
        (0, 1e5**2),
        (0**2, 6**2),
        (6**2, 12**2),
        (12**2, 1e5**2),
    ], "Areas are not set correctly"


def test_update_with_numpy_array(setup_panoptic_quality):
    pq = setup_panoptic_quality

    preds = np.array([[[[3, 1], [4, 1]], [[7, 0], [8, 0]]]])
    targets = np.array([[[[3, 1], [4, 1]], [[7, 0], [8, 0]]]])

    pq.update(preds, targets)
    assert pq.metric.true_positives.sum() > 0, "True positives should be updated"


def test_update_with_torch_tensor(setup_panoptic_quality):
    pq = setup_panoptic_quality

    preds = torch.tensor([[[[3, 1], [4, 1]], [[7, 0], [8, 0]]]])
    targets = torch.tensor([[[[3, 1], [4, 1]], [[7, 0], [8, 0]]]])

    pq.update(preds, targets)
    assert pq.metric.true_positives.sum() > 0, "True positives should be updated"

def test_compute_metric(setup_panoptic_quality):
    pq = setup_panoptic_quality
    
    preds = np.array([[[[3, 1], [4, 1]], [[7, 0], [8, 0]]]])
    targets = np.array([[[[3, 1], [4, 1]], [[7, 0], [8, 0]]]])

    pq.update(preds, targets)
    result = pq.compute()
    assert isinstance(result, torch.Tensor), "Result should be a torch Tensor"
    assert result is not None, "Compute should return a valid result"


def test_update_and_compute(setup_panoptic_quality):
    pq = setup_panoptic_quality
    
    preds = np.array([[[[3, 1], [4, 1]], [[7, 0], [8, 0]]]])
    targets = np.array([[[[3, 1], [4, 1]], [[7, 0], [8, 0]]]])

    result = pq.update_and_compute(preds, targets)
    assert isinstance(result, torch.Tensor), "Result should be a torch Tensor"
    assert result is not None, "Update and compute should return a valid result"
    pq_1 = setup_panoptic_quality
    pq_1.update(preds, targets)
    assert result == pq_1.compute(), "Update and compute should return the same result as compute"

