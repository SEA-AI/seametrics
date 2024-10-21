"""
TL:DR I know that this file does not follow the name convention, but it causes
an error since there is already another file with this name.
"""

import fiftyone as fo
import numpy as np
import pytest

from seametrics.panoptic.utils import (
    multiple_masks_to_single_mask,
    payload_to_seg_metric,  # This functin is used by the HF pipeline
)
from seametrics.payload import Payload, Resolution, Sequence


def generate_synthetic_payload(
    dataset: bool = True,
    model: bool = True,
    gt_field_name: bool = True,
    gt_config: list = [],
    model_config: list = [],
):
    mock_detections = [
        fo.Detection(
            label="MOTORBOAT",
            bounding_box=[0.0, 0.0, 0.015625, 0.009765625],
            mask=np.array(
                [
                    [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                ],
                dtype=np.uint8,
            ),
        ),
        fo.Detection(
            id="6682d49a4cb7459c1be09c52",
            attributes={},
            tags=[],
            label="SPHERICAL_BUOY",
            bounding_box=[0.603125, 0.591796875, 0.0109375, 0.009765625],
            mask=np.array(
                [
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                ]
            ),
            confidence=None,
            index=5,
        ),
    ]

    payload = Payload(
        dataset="dataset" if dataset else None,
        models=["model"] if model else None,
        gt_field_name="ground_truth_det" if gt_field_name else None,
        sequences={
            "sequence_a": Sequence(
                resolution=Resolution(height=512, width=640),
                ground_truth_det=[
                    [mock_detections[i] for i in range(detections)]
                    for detections in gt_config
                ],
                model=[
                    [mock_detections[i] for i in range(detections)]
                    for detections in model_config
                ],
            )
        },
    )

    return payload


def test_multiple_masks_to_single_mask_empty_prediction():
    frame_dets = []
    h, w = 100, 100
    label2id = {}
    single_mask = multiple_masks_to_single_mask(frame_dets, h, w, label2id)
    assert single_mask.shape == (100, 100, 2)
    assert label2id == {}
    assert (single_mask[:, :, 0] == -1).sum() == 10000
    assert (single_mask[:, :, 1] == -1).sum() == 10000
    assert np.unique(single_mask[:, :, 0]).size == 1
    assert np.unique(single_mask[:, :, 1]).size == 1


def test_multiple_masks_to_single_mask_single_prediction():
    frame_dets = [
        fo.Detection(
            label="MOTORBOAT",
            bounding_box=[0.0, 0.0, 0.015625, 0.009765625],
            mask=np.array(
                [
                    [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                ],
                dtype=np.uint8,
            ),
            # Each mock mask is a 6x11 numpy array with values 0 or 1. They all sum up to 50 unless explicitly stated otherwise
        )
    ]
    h, w = 512, 640
    label2id = {"MOTORBOAT": 1}
    single_mask = multiple_masks_to_single_mask(frame_dets, h, w, label2id)
    assert single_mask.shape == (512, 640, 2)
    assert label2id == {"MOTORBOAT": 1}
    assert (
        (single_mask[:, :, 0] == -1).sum()
        == (single_mask[:, :, 1] == -1).sum()
        == 327630
    )
    assert (single_mask[:, :, 0] != -1).sum() == (single_mask[:, :, 0] == 1).sum() == 50
    assert (single_mask[:, :, 1] != -1).sum() == (single_mask[:, :, 1] == 0).sum() == 50
    assert np.unique(single_mask[:, :, 0]).size == 2
    assert np.unique(single_mask[:, :, 1]).size == 2


def test_multiple_masks_to_single_mask_no_label2id():
    frame_dets = [
        fo.Detection(
            label="MOTORBOAT",
            bounding_box=[0.5078125, 0.509765625, 0.015625, 0.009765625],
            mask=np.array(
                [
                    [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                ],
                dtype=np.uint8,
            ),
        )
    ]
    h, w = 512, 640
    label2id = {}
    single_mask = multiple_masks_to_single_mask(frame_dets, h, w, label2id)
    assert single_mask.shape == (512, 640, 2)
    assert label2id == {"MOTORBOAT": 0}
    assert (
        (single_mask[:, :, 0] == -1).sum()
        == (single_mask[:, :, 1] == -1).sum()
        == 327630
    )
    assert (single_mask[:, :, 0] != -1).sum() == (single_mask[:, :, 0] == 0).sum() == 50
    assert (single_mask[:, :, 1] != -1).sum() == (single_mask[:, :, 1] == 0).sum() == 50
    assert np.unique(single_mask[:, :, 0]).size == 2
    assert np.unique(single_mask[:, :, 1]).size == 2


def test_multiple_masks_to_single_mask_multiple_non_overlapping_masks():
    frame_dets = [
        fo.Detection(
            label="MOTORBOAT",
            bounding_box=[0.0, 0.0, 0.015625, 0.009765625],
            mask=np.array(
                [
                    [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                ],
                dtype=np.uint8,
            ),
        ),
        fo.Detection(
            label="MOTORBOAT",
            bounding_box=[0.5078125, 0.509765625, 0.015625, 0.009765625],
            mask=np.array(
                [
                    [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                ],
                dtype=np.uint8,
            ),
        ),
    ]
    h, w = 512, 640
    label2id = {}
    single_mask = multiple_masks_to_single_mask(frame_dets, h, w, label2id)
    assert single_mask.shape == (512, 640, 2)
    assert label2id == {"MOTORBOAT": 0}
    assert (
        (single_mask[:, :, 0] == -1).sum()
        == (single_mask[:, :, 1] == -1).sum()
        == 327580
    )
    assert (
        (single_mask[:, :, 0] != -1).sum() == (single_mask[:, :, 0] == 0).sum() == 100
    )
    assert (single_mask[:, :, 1] != -1).sum() == 100
    assert (single_mask[:, :, 1] == 0).sum() == 50
    assert (single_mask[:, :, 1] == 1).sum() == 50
    assert np.unique(single_mask[:, :, 0]).size == 2
    assert np.unique(single_mask[:, :, 1]).size == 3


def test_multiple_masks_to_single_mask_multiple_overlapping_masks():
    # Both masks are nearly identical, only the first 2 pixels are different
    # Both sum up to 50
    frame_dets = [
        fo.Detection(
            label="MOTORBOAT",
            bounding_box=[0.0, 0.0, 0.015625, 0.009765625],
            mask=np.array(
                [
                    [1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                ],
                dtype=np.uint8,
            ),
        ),
        fo.Detection(
            label="MOTORBOAT",
            bounding_box=[0.0, 0.0, 0.015625, 0.009765625],
            mask=np.array(
                [
                    [0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                ],
                dtype=np.uint8,
            ),
        ),
    ]
    h, w = 512, 640
    label2id = {}
    single_mask = multiple_masks_to_single_mask(frame_dets, h, w, label2id)
    assert single_mask.shape == (512, 640, 2)
    assert label2id == {"MOTORBOAT": 0}
    assert (
        (single_mask[:, :, 0] == -1).sum()
        == (single_mask[:, :, 1] == -1).sum()
        == 327629
    )
    assert (single_mask[:, :, 0] != -1).sum() == (single_mask[:, :, 0] == 0).sum() == 51
    assert (single_mask[:, :, 1] != -1).sum() == 51
    assert (single_mask[:, :, 1] == 0).sum() == 1
    assert (single_mask[:, :, 1] == 1).sum() == 50
    assert np.unique(single_mask[:, :, 0]).size == 2
    assert np.unique(single_mask[:, :, 1]).size == 3


def test_payload_to_seg_metric_mock_payload():
    payload = generate_synthetic_payload(gt_config=[1, 2], model_config=[1, 2])
    pred, gt, label2id = payload_to_seg_metric(payload=payload, model_name="model")
    assert pred.shape == (2, 512, 640, 2)
    assert gt.shape == (2, 512, 640, 2)
    assert label2id == {"MOTORBOAT": 0, "SPHERICAL_BUOY": 1}
    assert (pred == gt).all()
    assert np.unique(pred).size == 3
    assert np.unique(gt).size == 3


def test_payload_to_seg_metric_empty_payload():
    payload = {}
    model_name = "model"
    label2id = {"WATER": 0, "SKY": 1}

    with pytest.raises(AttributeError):
        _, _, label2id = payload_to_seg_metric(payload, model_name, label2id)

    assert label2id == {"WATER": 0, "SKY": 1}


def test_payload_to_seg_metric_missing_model_name():
    payload = generate_synthetic_payload(gt_config=[1], model_config=[1])
    with pytest.raises(TypeError):
        payload_to_seg_metric(payload=payload)


def test_payload_to_seg_metric_model_name_not_on_sequence():
    payload = generate_synthetic_payload(gt_config=[1], model_config=[1])
    with pytest.raises(AttributeError):
        payload_to_seg_metric(payload=payload, model_name="not_model")


def test_payload_to_seg_metric_missing_gt_field():
    payload = generate_synthetic_payload(
        gt_field_name=False, gt_config=[1], model_config=[1]
    )
    with pytest.raises(AttributeError):
        payload_to_seg_metric(payload=payload, model_name="model")


def test_payload_to_seg_metric_label2id_populated():
    payload = generate_synthetic_payload(gt_config=[1], model_config=[2])
    label2id = {"WATER": 0, "SKY": 1}
    _, _, label2id = payload_to_seg_metric(
        payload=payload, model_name="model", label2id=label2id
    )
    assert label2id == {'WATER': 0, 'SKY': 1, 'MOTORBOAT': 2, 'SPHERICAL_BUOY': 3}


def test_payload_to_seg_metric_none_label2id():
    payload = generate_synthetic_payload(gt_config=[1], model_config=[2])
    label2id = None
    pred, gt, label2id = payload_to_seg_metric(
        payload=payload, model_name="model", label2id=label2id
    )
    assert label2id == {"MOTORBOAT": 0, "SPHERICAL_BUOY": 1}

def test_payload_to_seg_metric_single_frame():
    payload = generate_synthetic_payload(gt_config=[1], model_config=[1])
    pred, gt, label2id = payload_to_seg_metric(payload=payload, model_name="model")
    assert label2id == {'MOTORBOAT': 0}
    assert pred.shape == (1, 512, 640, 2)
    assert gt.shape == (1, 512, 640, 2)
    assert (pred == gt).all()
    assert np.unique(pred).size == 2
    assert np.unique(gt).size == 2


def test_payload_to_seg_metric_partial_gt_and_preds():
    payload = generate_synthetic_payload(gt_config=[1, 2, 0], model_config=[1, 0, 2])
    pred, gt, label2id = payload_to_seg_metric(payload=payload, model_name="model")
    assert label2id == {'MOTORBOAT': 0, 'SPHERICAL_BUOY': 1}
    assert pred.shape == (3, 512, 640, 2)
    assert gt.shape == (3, 512, 640, 2)
    assert (pred != gt).any()
    assert pred[0,:,:,0].sum() == pred[0,:,:,1].sum() == -327630
    assert pred[1,:,:,0].sum() == pred[1,:,:,1].sum() == -327680
    assert pred[2,:,:,0].sum() == pred[2,:,:,1].sum() == -327560
    assert gt[0,:,:,0].sum() == gt[0,:,:,1].sum() == -327630
    assert gt[1,:,:,0].sum() == gt[1,:,:,1].sum() == -327560
    assert gt[2,:,:,0].sum() == gt[2,:,:,1].sum() == -327680
    assert np.unique(pred).size == 3
    assert np.unique(gt).size == 3
    assert np.unique(pred[0,:,:,0]).size == np.unique(gt[0,:,:,0]).size == np.unique(pred[0,:,:,1]).size == np.unique(gt[0,:,:,1]).size == 2
    assert np.unique(pred[2,:,:,0]).size == np.unique(gt[1,:,:,0]).size == np.unique(pred[2,:,:,1]).size == np.unique(gt[1,:,:,1]).size == 3
    assert np.unique(pred[1,:,:,0]).size == np.unique(gt[2,:,:,0]).size == np.unique(pred[1,:,:,1]).size == np.unique(gt[2,:,:,1]).size == 1
