import fiftyone as fo
import numpy as np
import pytest

from seametrics.detection import PrecisionRecallF1Support
from seametrics.detection.pooling import aggregate_sequence_results
from seametrics.detection.utils import (
    _validate_arrays,
    frame_dets_to_det_metrics,
    get_relevant_fields,
    get_values,
    payload_sequence_to_det_metrics,
    payload_to_det_metric,
    payload_to_det_metrics_by_sequence,
    prepare_data_for_det_metrics,
    sequence_results_to_df,
)
from seametrics.payload import Payload, Resolution, Sequence


def assert_det_dicts(dict1, dict2):
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())
    assert keys1 == keys2, f"Keys mismatch: {keys1 ^ keys2}"

    for key in dict1:
        assert np.array_equal(dict1[key], dict2[key]), f"Value mismatch for key {key}"


def assert_list_of_det_dicts(list1, list2):
    assert len(list1) == len(list2), "Length mismatch"
    for dict1, dict2 in zip(list1, list2, strict=True):
        assert_det_dicts(dict1, dict2)


def test_frame_dets_to_det_metrics_with_empty_predictions():
    dets = []
    w = 640
    h = 512
    desired_format = {
        "boxes": np.array([]),
        "scores": np.array([]),
        "labels": np.array([]),
    }
    metrics_format = frame_dets_to_det_metrics(dets, w, h)
    assert_det_dicts(metrics_format, desired_format)


def test_frame_dets_to_det_metrics_with_empty_ground_truth():
    dets = []
    w = 640
    h = 512
    desired_format = {
        "boxes": np.array([]),
        "labels": np.array([]),
    }
    metrics_format = frame_dets_to_det_metrics(dets, w, h, is_gt=True)
    # check both dicts have the same keys
    assert_det_dicts(metrics_format, desired_format)


def test_frame_dets_to_det_metrics_with_detections():
    dets = [
        fo.Detection(
            label="FAR_AWAY_OBJECT",
            bounding_box=[0, 0, 1, 1],
            confidence=0.153076171875,
        )
    ]
    w = 640
    h = 512
    desired_format = {
        "boxes": np.array([[0, 0, 640, 512]]),
        "scores": np.array([0.153076171875]),
        "labels": np.array([0]),
    }
    metrics_format = frame_dets_to_det_metrics(dets, w, h)
    # check both dicts have the same keys
    assert_det_dicts(metrics_format, desired_format)


def test_frame_dets_to_det_metrics_with_ground_truth():
    dets = [
        fo.Detection(
            label="FAR_AWAY_OBJECT",
            bounding_box=[0, 0, 1, 1],
            area=100,  # not forwarded: the metric derives area from the box
        ),
        fo.Detection(
            label="FAR_AWAY_OBJECT",
            bounding_box=[0, 0, 0.5, 0.5],
            area=50,  # not forwarded: the metric derives area from the box
        ),
    ]
    w = 640
    h = 512
    desired_format = {
        "boxes": np.array(
            [
                [0, 0, 640, 512],
                [0, 0, 320, 256],
            ]
        ),
        "labels": np.array([0, 0]),
    }
    metrics_format = frame_dets_to_det_metrics(dets, w, h, is_gt=True)
    # check both dicts have the same keys
    assert_det_dicts(metrics_format, desired_format)


def test_payload_sequence_to_det_metrics_with_empty_sequence():
    sequence_dets = []
    w = 640
    h = 512
    desired_output = []
    output = payload_sequence_to_det_metrics(sequence_dets, w, h)
    assert_list_of_det_dicts(output, desired_output)


def test_payload_sequence_to_det_metrics_with_empty_predictions():
    sequence_dets = [[]]
    w = 640
    h = 512
    desired_outputs = [
        {
            "boxes": np.array([]),
            "scores": np.array([]),
            "labels": np.array([]),
        }
    ]
    outputs = payload_sequence_to_det_metrics(sequence_dets, w, h)
    assert_list_of_det_dicts(outputs, desired_outputs)


def test_payload_sequence_to_det_metrics_with_empty_ground_truth():
    sequence_dets = [[]]
    w = 640
    h = 512
    desired_outputs = [
        {
            "boxes": np.array([]),
            "labels": np.array([]),
        }
    ]
    outputs = payload_sequence_to_det_metrics(sequence_dets, w, h, is_gt=True)
    assert_list_of_det_dicts(outputs, desired_outputs)


def test_payload_sequence_to_det_metrics_with_detections():
    sequence_dets = [
        [
            fo.Detection(
                label="FAR_AWAY_OBJECT",
                bounding_box=[0, 0, 1, 1],
                confidence=0.153076171875,
            )
        ]
    ] * 10
    w = 640
    h = 512
    desired_outputs = [
        {
            "boxes": np.array([[0, 0, 640, 512]]),
            "scores": np.array([0.153076171875]),
            "labels": np.array([0]),
        }
        for _ in range(10)
    ]
    outputs = payload_sequence_to_det_metrics(sequence_dets, w, h)
    assert_list_of_det_dicts(outputs, desired_outputs)


def test_payload_sequence_to_det_metrics_with_ground_truth():
    sequence_dets = [
        [
            fo.Detection(
                label="FAR_AWAY_OBJECT",
                bounding_box=[0, 0, 1, 1],
                area=100,  # not forwarded: the metric derives area from the box
            ),
            fo.Detection(
                label="FAR_AWAY_OBJECT",
                bounding_box=[0, 0, 0.5, 0.5],
                area=50,  # not forwarded: the metric derives area from the box
            ),
        ]
    ] * 10
    w = 640
    h = 512
    desired_outputs = [
        {
            "boxes": np.array(
                [
                    [0, 0, 640, 512],
                    [0, 0, 320, 256],
                ]
            ),
            "labels": np.array([0, 0]),
        }
        for _ in range(10)
    ]
    outputs = payload_sequence_to_det_metrics(sequence_dets, w, h, is_gt=True)
    assert_list_of_det_dicts(outputs, desired_outputs)


def test_prepare_data_for_det_metrics_denormalizes_and_converts():
    # normalized xywh (0.1, 0.2, 0.3, 0.4) on 100x50 -> xywh (10, 10, 30, 20)
    # -> xyxy (10, 10, 40, 30)
    gt = [[[0.1, 0.2, 0.3, 0.4]], [[0.5, 0.5, 0.1, 0.2]]]
    dt = [[[0.1, 0.2, 0.3, 0.4]], []]
    target, preds = prepare_data_for_det_metrics(
        gt, [["boat"], ["buoy"]], dt, [["boat"], []], [[None], []], 100, 50
    )
    assert_list_of_det_dicts(
        target,
        [
            {"boxes": np.array([[10.0, 10.0, 40.0, 30.0]]), "labels": np.array([0])},
            {"boxes": np.array([[50.0, 25.0, 60.0, 35.0]]), "labels": np.array([0])},
        ],
    )
    # all scores None -> every prediction scored 1.0
    assert_list_of_det_dicts(
        preds,
        [
            {
                "boxes": np.array([[10.0, 10.0, 40.0, 30.0]]),
                "labels": np.array([0]),
                "scores": np.array([1.0]),
            },
            {"boxes": np.array([]), "labels": np.array([]), "scores": np.array([])},
        ],
    )


class _View:
    media_type = "group"


@pytest.mark.parametrize("fn", [get_relevant_fields, get_values])
def test_fiftyone_helpers_reject_unknown_media_type(fn):
    with pytest.raises(ValueError, match="Unsupported media type: group"):
        fn(_View(), "gt")


def test_validate_arrays_wraps_a_single_frame():
    # one frame of two boxes becomes a one-frame list
    frames = _validate_arrays([[0, 0, 1, 1], [0, 0, 2, 2]], data_type="bbox")
    assert len(frames) == 1
    np.testing.assert_array_equal(frames[0], [[0, 0, 1, 1], [0, 0, 2, 2]])
    assert _validate_arrays([0.5, 0.7], data_type="score")[0].tolist() == [0.5, 0.7]
    assert _validate_arrays(None, data_type="label")[0].size == 0


def test_validate_arrays_rejects_unknown_data_type():
    with pytest.raises(ValueError, match="Unsupported data type: keypoint"):
        _validate_arrays([[0, 0, 1, 1]], data_type="keypoint")


# On a 100x100 image these boxes have IoU exactly 1/3:
#   gt   rel [0.1, 0.1, 0.2, 0.2] -> px xywh [10, 10, 20, 20]
#   pred rel [0.2, 0.1, 0.2, 0.2] -> px xywh [20, 10, 20, 20]
#   intersection 10 * 20 = 200, union 400 + 400 - 200 = 600
GT_BOX = [0.1, 0.1, 0.2, 0.2]
PRED_BOX = [0.2, 0.1, 0.2, 0.2]
# split at 300 px², so the 20x20 px (400 px²) box is "large"
AREA_KWARGS = {
    "area_ranges": [[0, 300], [300, 1e10]],
    "area_ranges_labels": ["small", "large"],
}
LABEL_MAPPING = {"BOAT": 1, "BUOY": 2}


def _det(bbox=GT_BOX, label="BOAT", **kwargs: object):
    return fo.Detection(label=label, bounding_box=list(bbox), **kwargs)


def _pred(bbox=GT_BOX, label="BOAT"):
    return _det(bbox, label, confidence=0.9)


def _sequence(gt, pred, width=100, height=100, gt_field="gt", **fields: object):
    return Sequence(
        resolution=Resolution(height=height, width=width),
        **{gt_field: gt, "model": pred, **fields},
    )


def _payload(sequences, models=("model",), gt_field="gt"):
    return Payload(
        dataset="synthetic",
        models=list(models),
        gt_field_name=gt_field,
        sequences=sequences,
    )


def _compute(preds, refs, **kwargs: object):
    metric = PrecisionRecallF1Support(
        **{"box_format": "xywh", "iou_thresholds": [0.5], **kwargs}
    )
    metric.update(preds, refs)
    return metric.compute()["metrics"]


def _by_sequence(payload, **kwargs: object):
    return payload_to_det_metrics_by_sequence(
        payload, **{"iou_thresholds": [0.5], **kwargs}
    )


def _overall(per_sequence_results):
    return aggregate_sequence_results(per_sequence_results)["metrics"]


def _hit_and_miss():
    return _payload(
        {
            "hit": _sequence([[_det()]], [[_pred()]]),
            "miss": _sequence([[_det()]], [[]]),
        }
    )


@pytest.mark.parametrize(
    ("bbox", "w", "h", "expected"),
    [
        pytest.param([0.25, 0.5, 0.5, 0.25], 640, 512, [160, 256, 320, 128], id="xywh"),
        pytest.param([0.1, 0.2, 0.3, 0.4], 400, 100, [40, 20, 120, 40], id="wide"),
        pytest.param([0.1, 0.2, 0.3, 0.4], 100, 400, [10, 80, 30, 160], id="tall"),
        pytest.param(
            [0.9, 0.9, 0.5, 0.5], 100, 200, [90, 180, 50, 100], id="unclipped"
        ),
    ],
)
def test_frame_dets_scale_relative_xywh_to_pixels(bbox, w, h, expected):
    """x-like values scale by width, y-like by height; the layout stays xywh."""
    out = frame_dets_to_det_metrics([_pred(bbox)], w, h)
    np.testing.assert_allclose(out["boxes"], [expected])


def test_frame_dets_scores_keep_zero_and_default_missing_to_one():
    dets = [_det(confidence=0.0), _det(), _det(confidence=0.75)]
    out = frame_dets_to_det_metrics(dets, 100, 100)
    np.testing.assert_array_equal(out["scores"], [0.0, 1.0, 0.75])


def test_frame_dets_unmapped_labels_dropped_keeping_alignment():
    dets = [
        _det([0.0, 0.0, 0.1, 0.1], confidence=0.9),
        _det([0.1, 0.1, 0.1, 0.1], "UNKNOWN", confidence=0.1),
        _det([0.2, 0.2, 0.1, 0.1], confidence=0.5),
    ]
    out = frame_dets_to_det_metrics(dets, 100, 100, label_mapping={"BOAT": 1})
    np.testing.assert_array_equal(out["labels"], [1, 1])
    np.testing.assert_array_equal(out["scores"], [0.9, 0.5])
    np.testing.assert_allclose(out["boxes"][:, 0], [0.0, 20.0])


@pytest.mark.parametrize(
    ("w", "h", "gt", "pred", "threshold", "counts"),
    [
        pytest.param(100, 100, GT_BOX, PRED_BOX, 0.3, (1, 0, 0), id="square-match"),
        pytest.param(100, 100, GT_BOX, PRED_BOX, 0.35, (0, 1, 1), id="square-miss"),
    ],
)
def test_converted_boxes_give_the_hand_computed_iou(w, h, gt, pred, threshold, counts):
    preds = payload_sequence_to_det_metrics([[_pred(pred)]], w, h)
    refs = payload_sequence_to_det_metrics([[_det(gt)]], w, h, is_gt=True)
    res = _compute(preds, refs, iou_thresholds=[threshold])["all"]
    assert (res["tp"], res["fp"], res["fn"]) == counts


def test_converted_boxes_need_box_format_xywh():
    """Read as xyxy, the prediction [20, 10, 20, 20] has zero width: no match."""
    preds = payload_sequence_to_det_metrics([[_pred(PRED_BOX)]], 100, 100)
    refs = payload_sequence_to_det_metrics([[_det()]], 100, 100, is_gt=True)
    xyxy = _compute(preds, refs, iou_thresholds=[0.3], box_format="xyxy")
    assert xyxy["all"]["tp"] == 0


def test_area_bucket_comes_from_the_box_not_an_annotated_area():
    """A 400 px² box stays "large" even when fo and the target dict say 100."""
    preds = payload_sequence_to_det_metrics([[_pred()]], 100, 100)
    refs = payload_sequence_to_det_metrics([[_det(area=100.0)]], 100, 100, is_gt=True)
    refs[0]["area"] = np.array([100.0])
    metrics = _compute(preds, refs, **AREA_KWARGS)
    assert (metrics["small"]["tp"], metrics["large"]["tp"]) == (0, 1)


def test_two_frames_one_hit_one_duplicate_one_miss():
    preds = payload_sequence_to_det_metrics([[_pred(), _pred()], []], 100, 100)
    refs = payload_sequence_to_det_metrics([[_det()], [_det()]], 100, 100, is_gt=True)
    res = _compute(preds, refs)["all"]
    assert (res["tp"], res["fp"], res["fn"], res["duplicates"]) == (1, 1, 1, 1)
    assert res["precision"] == res["recall"] == res["f1"] == pytest.approx(0.5)


def test_payload_to_det_metric_scales_each_sequence_by_its_resolution():
    payload = _payload(
        {
            "a": _sequence([[_det()]], [[_pred()]], width=200, height=100),
            "b": _sequence([[_det()]], [[_pred()]], width=400, height=50),
        }
    )
    preds, refs = payload_to_det_metric(payload, model_name="model")
    np.testing.assert_allclose(preds[0]["boxes"], [[20, 10, 40, 20]])
    np.testing.assert_allclose(preds[1]["boxes"], [[40, 5, 80, 10]])
    np.testing.assert_allclose(refs[1]["boxes"], preds[1]["boxes"])


def test_payload_to_det_metric_applies_label_mapping():
    preds, refs = payload_to_det_metric(
        _payload({"a": _sequence([[_det()]], [[_pred()]])}),
        label_mapping={"BOAT": 4},
        class_agnostic=False,
    )
    assert preds[0]["labels"].tolist() == refs[0]["labels"].tolist() == [4]


@pytest.mark.parametrize(
    ("fn", "kwargs", "match"),
    [
        (payload_to_det_metric, {"label_mapping": {"BOAT": 1}}, "Label mapping"),
        (_by_sequence, {"label_mapping": {"BOAT": 1}}, "Label mapping"),
        (_by_sequence, {"box_format": "xyxy"}, "box_format` cannot be overridden"),
    ],
)
def test_invalid_conversion_arguments_raise(fn, kwargs, match):
    with pytest.raises(ValueError, match=match):
        fn(_hit_and_miss(), **kwargs)


def test_by_sequence_scores_each_sequence(capsys):
    """Pooled recall 0.5 would hide that one sequence scored 1.0 and the other 0.0."""
    res = _by_sequence(_hit_and_miss())
    assert not capsys.readouterr().out  # no summary table per sequence
    hit, miss = (res[k]["metrics"]["all"] for k in res)
    assert list(res) == ["hit", "miss"]
    assert (hit["tp"], hit["fn"], hit["recall"]) == (1, 0, 1.0)
    assert (miss["tp"], miss["fn"], miss["recall"], miss["precision"]) == (0, 1, 0, -1)
    assert list(sequence_results_to_df(res)["n_imgs"]) == [1, 1]


def test_by_sequence_forwards_metric_kwargs():
    res = _by_sequence(_hit_and_miss(), **AREA_KWARGS)
    assert all(set(r["metrics"]) == {"small", "large"} for r in res.values())
    assert res["hit"]["metrics"]["large"]["tp"] == 1


@pytest.mark.parametrize(("model_name", "tp"), [(None, 1), ("model_b", 0)])
def test_by_sequence_model_defaults_to_the_first(model_name, tp):
    sequence = _sequence([[_det()]], [[_pred()]], model_b=[[]])
    payload = _payload({"s": sequence}, models=("model", "model_b"))
    assert (
        _by_sequence(payload, model_name=model_name)["s"]["metrics"]["all"]["tp"] == tp
    )


def test_by_sequence_reads_ground_truth_from_gt_field_name():
    sequence = _sequence([[_det()]], [[]], gt_field="ground_truth")
    payload = _payload({"s": sequence}, gt_field="ground_truth")
    assert _by_sequence(payload)["s"]["metrics"]["all"]["support"] == 1


def test_by_sequence_on_no_sequences_is_empty():
    assert _by_sequence(_payload({})) == {}


@pytest.mark.parametrize("area_kwargs", [{}, AREA_KWARGS], ids=["all", "area-ranges"])
def test_overall_equals_one_evaluation_of_every_frame(area_kwargs):
    """seq_a: a match, then a prediction on an empty frame (tp 1, fp 1, fpi 1).

    seq_b: a miss, then a misplaced prediction (fp 1, fn 2).
    """
    payload = _payload(
        {
            "seq_a": _sequence([[_det()], []], [[_pred()], [_pred()]]),
            "seq_b": _sequence([[_det()], [_det()]], [[], [_pred(PRED_BOX)]]),
        }
    )
    overall = _overall(_by_sequence(payload, **area_kwargs))
    assert overall == _compute(*payload_to_det_metric(payload), **area_kwargs)


def _disjoint_classes():
    return _payload(
        {
            "boat": _sequence([[_det()]], [[_pred()]]),
            "buoy": _sequence([[_det(label="BUOY")]], [[_pred(label="BUOY")]]),
        }
    )


def test_class_specific_sequences_report_every_mapped_class():
    """Each sequence reports both classes, so pooling sums them by position."""
    payload = _disjoint_classes()
    kwargs = {"label_mapping": LABEL_MAPPING, "class_agnostic": False}
    res = _by_sequence(payload, **kwargs)
    tps = [res[k]["metrics"]["all"]["tp"].tolist() for k in res]
    assert tps == [[1, 0], [0, 1]]
    pooled = _compute(
        *payload_to_det_metric(payload, **kwargs), class_agnostic=False, labels=[1, 2]
    )["all"]
    overall = _overall(res)["all"]
    for key in ("tp", "fp", "fn", "precision", "recall", "f1", "support"):
        np.testing.assert_allclose(overall[key], pooled[key], err_msg=key)


def test_class_specific_explicit_labels_are_kept():
    res = _by_sequence(
        _disjoint_classes(),
        label_mapping=LABEL_MAPPING,
        class_agnostic=False,
        labels=[1, 2, 3],
    )
    assert np.shape(res["boat"]["metrics"]["all"]["tp"]) == (3,)


def test_class_specific_without_mapping_pools_an_empty_sequence():
    """Unmapped labels are all 0, so a sequence with no boxes still reports it."""
    payload = _payload(
        {
            "hit": _sequence([[_det()]], [[_pred()]]),
            "empty": _sequence([[]], [[]]),
        }
    )
    overall = _overall(_by_sequence(payload, class_agnostic=False))["all"]
    assert (np.sum(overall["tp"]), np.sum(overall["fn"])) == (1, 0)


def _keyframe_sequence(pred=None, mask=(True, False, True, False)):
    """GT on four frames; the model emits on, and flags, frames 0 and 2."""
    return _sequence(
        [[_det()]] * 4,
        pred or [[_pred()], [], [_pred()], []],
        keyframes={"model": list(mask)},
    )


@pytest.mark.parametrize(
    ("keyframes_only", "expected"), [(False, (2, 0, 2, 4)), (True, (2, 0, 0, 2))]
)
def test_keyframes_only_drops_other_frames_from_both_sides(keyframes_only, expected):
    """GT on skipped frames is a miss unless keyframes_only drops those frames."""
    res = _by_sequence(
        _payload({"s": _keyframe_sequence()}), keyframes_only=keyframes_only
    )["s"]["metrics"]["all"]
    assert (res["tp"], res["fp"], res["fn"], res["nImgs"]) == expected


def test_keyframes_only_ignores_predictions_on_dropped_frames():
    pred = [[_pred()], [_pred(PRED_BOX)], [_pred()], []]
    payload = _payload({"s": _keyframe_sequence(pred=pred)})
    res = _by_sequence(payload, keyframes_only=True)["s"]["metrics"]["all"]
    assert (res["tp"], res["fp"], res["fn"]) == (2, 0, 0)


@pytest.mark.parametrize(
    ("sequence", "match"),
    [
        pytest.param(_sequence([[_det()]], [[]]), "no keyframe data", id="no-mask"),
        pytest.param(
            _sequence([[_det()]], [[]], keyframes={"other": [True]}),
            "no keyframe data",
            id="other-model",
        ),
        pytest.param(
            _keyframe_sequence(mask=(True, False)), "has 2 entries", id="length"
        ),
    ],
)
def test_keyframes_only_rejects_a_missing_or_misaligned_mask(sequence, match):
    with pytest.raises(ValueError, match=match):
        _by_sequence(_payload({"s": sequence}), keyframes_only=True)


def test_keyframes_only_overall_equals_the_pooled_route():
    payload = _payload({"a": _keyframe_sequence(), "b": _keyframe_sequence()})
    overall = _overall(_by_sequence(payload, keyframes_only=True))
    assert overall == _compute(*payload_to_det_metric(payload, keyframes_only=True))
