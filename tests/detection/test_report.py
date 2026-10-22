"""Tests for seametrics.detection.report."""

import warnings

import fiftyone as fo
import pytest

from seametrics.detection.pooling import OVERALL_KEY
from seametrics.detection.report import (
    DetectionReport,
    EvaluationOptions,
    _pool_comparable,
    evaluate_models,
)

BOX = [0.1, 0.1, 0.2, 0.2]
FAR = [0.7, 0.7, 0.1, 0.1]
OPTIONS = EvaluationOptions(iou_threshold=1e-9)


def _gt(box=BOX):
    return fo.Detection(label="BOAT", bounding_box=box)


def _pred(box=BOX, confidence=0.9, label="BOAT"):
    return fo.Detection(label=label, bounding_box=box, confidence=confidence)


def _result(tp):
    """A one-image, class-agnostic compute() output with *tp* hits."""
    counts = {"tp": tp, "fp": 0, "fn": 0, "duplicates": 0, "fpi": 0, "nImgs": 1}
    ratios = {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": tp}
    metric = {"range": [0, 1e10], "iouThr": "1e-09", "maxDets": 100}
    return {"metrics": {"all": {**metric, **counts, **ratios}}}


@pytest.mark.parametrize(
    ("per_sequence", "pooled", "overall_tp"),
    [
        pytest.param(
            {"a": {"s1": _result(1), "s2": _result(2)}, "b": {"s1": _result(3)}},
            ["s1"],
            {"a": 1, "b": 3},
            id="intersection-only",
        ),
        pytest.param(
            {"a": {"s1": _result(1)}, "b": {"s2": _result(1)}}, [], {}, id="disjoint"
        ),
        pytest.param(
            {"a": {"s1": _result(1), "s2": _result(2)}},
            ["s1", "s2"],
            {"a": 3},
            id="one",
        ),
    ],
)
def test_pool_comparable_pools_the_sequences_every_model_has(
    per_sequence, pooled, overall_tp
):
    overall, shared = _pool_comparable(per_sequence, list(per_sequence))
    assert shared == pooled
    assert {m: r["metrics"]["all"]["tp"] for m, r in overall.items()} == overall_tp


def test_report_df_flags_pooled_rows_and_lists_exclusions():
    report = DetectionReport(
        models=["model"],
        per_sequence={"model": {"seq": _result(1), "dropped": _result(0)}},
        overall={"model": _result(1)},
        pooled_sequences=["seq"],
        skipped={"model": ["other"]},
    )
    df = report.to_df()
    assert list(df["sequence"]) == ["seq", "dropped", OVERALL_KEY]
    assert list(df["in_overall"]) == [True, False, False]
    assert report.excluded_sequences == ["other"]


def _video(dataset, sequence, frames):
    """Add one video sample; *frames* maps frame number to {field: Detections}."""
    sample = fo.Sample(
        filepath=f"{sequence}.mp4",
        sequence=sequence,
        metadata=fo.VideoMetadata(frame_width=100, frame_height=100),
    )
    for number, fields in frames.items():
        sample.frames[number] = fo.Frame(**fields)
    dataset.add_sample(sample)


def _hit_and_strays(dataset):
    """Frame 1: a cross-class hit and a stray. Frame 2: no ground truth, a stray."""
    _video(
        dataset,
        "seq-1",
        {
            1: {
                "gt": fo.Detections(detections=[_gt()]),
                "model": fo.Detections(
                    detections=[_pred(label="SHIP"), _pred(FAR, 0.5)]
                ),
            },
            2: {
                "gt": fo.Detections(detections=[]),
                "model": fo.Detections(detections=[_pred(FAR, 0.4)]),
            },
        },
    )


def test_eval_key_writes_a_fiftyone_evaluation_matching_the_metrics(dataset):
    _hit_and_strays(dataset)
    evaluate_models(dataset.name, "gt", ["model"], options=OPTIONS)
    assert dataset.list_evaluations() == []

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        for _ in range(2):  # a rerun replaces the evaluation
            report = evaluate_models(
                dataset.name,
                "gt",
                ["model"],
                options=OPTIONS,
                eval_key="ev",
            )

    summary = report.per_sequence["model"]["seq-1"]["metrics"]["all"]
    assert (summary["tp"], summary["fp"]) == (1, 2)
    assert dataset.list_evaluations() == ["ev_model"]
    assert dataset.values("frames.model.detections.ev_model") == [
        [["tp", "fp"], ["fp"]]
    ]
    assert dataset.values("frames.gt.detections.ev_model") == [[["tp"], []]]


def test_eval_key_warns_when_fiftyone_disagrees_with_the_report(dataset):
    """An "all" range that excludes the box makes the report count no hit."""
    _hit_and_strays(dataset)
    with pytest.warns(UserWarning, match="counts tp, fp"):
        evaluate_models(
            dataset.name,
            "gt",
            ["model"],
            options=OPTIONS,
            eval_key="ev",
            area_ranges=[[0, 10]],
            area_ranges_labels=["all"],
        )


@pytest.mark.filterwarnings("error::UserWarning")
def test_keyframes_only_skips_a_model_without_flags_and_pools_the_rest(dataset):
    """Model "b" flags no keyframe on seq-2, so seq-2 leaves the pool for both."""
    # seq-1 frame 2 is a missed object on a non-keyframe: fn only if evaluated
    _video(
        dataset,
        "seq-1",
        {
            frame: {
                "gt": fo.Detections(detections=[_gt()]),
                **{
                    model: fo.Detections(detections=preds, keyframe=keyframe)
                    for model in ("a", "b")
                },
            }
            for frame, keyframe, preds in ((1, True, [_pred()]), (2, False, []))
        },
    )
    _video(
        dataset,
        "seq-2",
        {
            1: {
                "gt": fo.Detections(detections=[_gt()]),
                "a": fo.Detections(detections=[_pred()], keyframe=True),
                "b": fo.Detections(detections=[_pred()]),
            }
        },
    )

    report = evaluate_models(
        dataset.name,
        "gt",
        ["a", "b"],
        options=EvaluationOptions(keyframes_only=True, iou_threshold=1e-9),
        eval_key="ev",
    )

    assert report.skipped == {"a": [], "b": ["seq-2"]}
    assert report.pooled_sequences == ["seq-1"]
    assert sorted(report.per_sequence["a"]) == ["seq-1", "seq-2"]
    for model in ("a", "b"):
        pooled = report.overall[model]["metrics"]["all"]
        assert (pooled["tp"], pooled["fn"], pooled["nImgs"]) == (1, 0, 1)
    # FiftyOne skips the non-keyframe too: its missed object is not marked fn
    assert dataset.values("frames.gt.detections.ev_a") == [[["tp"], [None]], [["tp"]]]


@pytest.mark.parametrize(
    ("models", "eval_key", "match"),
    [([], None, "at least one model"), (["pred-v2"], "ev", "not an identifier")],
)
def test_evaluate_models_rejects_bad_arguments_before_loading(models, eval_key, match):
    with pytest.raises(ValueError, match=match):
        evaluate_models("unused", "gt", models, eval_key=eval_key)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"iou_threshold": 0}, "iou_threshold must be > 0"),
        ({"iou_threshold": -0.1}, "iou_threshold must be > 0"),
        ({"label_mapping": {"BOAT": 1}}, "class_agnostic=False"),
    ],
)
def test_invalid_options_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        EvaluationOptions(**kwargs)


def test_evaluation_kwargs_carry_every_option():
    options = EvaluationOptions(
        keyframes_only=True,
        iou_threshold=0.5,
        label_mapping={"BOAT": 1},
        class_agnostic=False,
    )
    assert options.evaluation_kwargs("m", {"area_ranges": [[0, 1]]}) == {
        "model_name": "m",
        "label_mapping": {"BOAT": 1},
        "class_agnostic": False,
        "keyframes_only": True,
        "iou_thresholds": [0.5],
        "area_ranges": [[0, 1]],
    }


def test_options_cannot_be_overridden_through_metric_kwargs():
    with pytest.raises(ValueError, match="iou_thresholds"):
        OPTIONS.evaluation_kwargs("m", {"iou_thresholds": [0]})


def test_empty_sequence_list_evaluates_nothing(dataset):
    _video(dataset, "seq-1", {1: {"gt": fo.Detections(detections=[_gt()])}})
    report = evaluate_models(dataset.name, "gt", ["model"], sequence_list=[])
    assert report.per_sequence == {"model": {}}


def test_a_requested_sequence_that_is_missing_is_excluded(dataset):
    _hit_and_strays(dataset)
    report = evaluate_models(
        dataset.name, "gt", ["model"], sequence_list=["seq-1", "typo"]
    )
    assert sorted(report.per_sequence["model"]) == ["seq-1"]
    assert report.excluded_sequences == ["typo"]
    assert report.pooled_sequences == ["seq-1"]


@pytest.mark.parametrize("slices", [None, ["wide", "narrow"]])
def test_a_group_dataset_needs_exactly_one_slice(dataset, slices):
    """Slices share sequence names, so two would overwrite each other."""
    dataset.add_group_field("group", default="wide")
    group = fo.Group()
    dataset.add_samples(
        [
            fo.Sample(filepath=f"{s}.mp4", group=group.element(s))
            for s in ("wide", "narrow")
        ]
    )
    with pytest.raises(ValueError, match="one slice"):
        evaluate_models(dataset.name, "gt", ["model"], slices=slices)


def test_only_the_chosen_slice_is_evaluated(dataset):
    """The wide slice hits the boat and the narrow one misses it."""
    dataset.add_group_field("group", default="wide")
    group = fo.Group()
    for name, preds in (("wide", [_pred()]), ("narrow", [])):
        sample = fo.Sample(
            filepath=f"{name}.mp4",
            group=group.element(name),
            sequence="seq",
            metadata=fo.VideoMetadata(frame_width=100, frame_height=100),
        )
        sample.frames[1] = fo.Frame(
            gt=fo.Detections(detections=[_gt()]),
            model=fo.Detections(detections=preds),
        )
        dataset.add_sample(sample)
    report = evaluate_models(dataset.name, "gt", ["model"], slices=["narrow"])
    summary = report.per_sequence["model"]["seq"]["metrics"]["all"]
    assert (summary["tp"], summary["fn"]) == (0, 1)
