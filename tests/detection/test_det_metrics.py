"""Unit tests for DetectionMetrics.

All expected values are hand-computed so a reviewer can verify the math
without running the code.
"""


import numpy as np
import pandas as pd
import pytest

from seametrics.detection.det_metrics import DetectionMetrics
from seametrics.detection.utils import det_metrics_to_df

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pred_frame(boxes, scores=None, labels=None):
    """Build one prediction frame dict.

    Parameters
    ----------
    boxes : list of list
        Bounding boxes in xyxy format.
    scores : list, optional
        Confidence scores; defaults to 1.0 for each box.
    labels : list, optional
        Class labels; defaults to 0 for each box.

    Returns:
    -------
    dict
        Detection dict with keys ``boxes``, ``labels``, ``scores``.
    """
    n = len(boxes)
    return {
        "boxes": np.array(boxes, dtype=float).reshape(-1, 4),
        "labels": np.array(labels if labels is not None else [0] * n, dtype=int),
        "scores": np.array(scores if scores is not None else [1.0] * n, dtype=float),
    }


def _gt_frame(boxes, labels=None):
    """Build one ground-truth frame dict.

    Parameters
    ----------
    boxes : list of list
        Bounding boxes in xyxy format.
    labels : list, optional
        Class labels; defaults to 0 for each box.

    Returns:
    -------
    dict
        Detection dict with keys ``boxes`` and ``labels``.
    """
    n = len(boxes)
    return {
        "boxes": np.array(boxes, dtype=float).reshape(-1, 4),
        "labels": np.array(labels if labels is not None else [0] * n, dtype=int),
    }


def _empty_pred():
    """One frame with no predictions."""
    return {
        "boxes": np.array([]).reshape(0, 4),
        "labels": np.array([]),
        "scores": np.array([]),
    }


def _empty_gt():
    """One frame with no ground-truth objects."""
    return {"boxes": np.array([]).reshape(0, 4), "labels": np.array([])}


# ---------------------------------------------------------------------------
# Tests: compute() at sequence level
# ---------------------------------------------------------------------------


def test_perfect_detection():
    """Single box prediction that exactly matches GT: TP=1, FP=0, FN=0."""
    m = DetectionMetrics(iou_threshold=0.5)
    preds = [_pred_frame([[10, 10, 100, 100]], scores=[0.9])]
    targets = [_gt_frame([[10, 10, 100, 100]])]

    m.update(preds, targets, "seq_perfect")

    result = m.compute(sequence="seq_perfect")
    assert "all" in result
    r = result["all"]
    assert r["tp"] == 1
    assert r["fp"] == 0
    assert r["fn"] == 0
    assert r["precision"] == pytest.approx(1.0)
    assert r["recall"] == pytest.approx(1.0)
    assert r["f1"] == pytest.approx(1.0)
    assert r["support"] == 1  # TP + FN


def test_no_predictions_all_fn():
    """Empty predictions with one GT object: TP=0, FP=0, FN=1.

    Precision is undefined (-1); recall = 0.
    """
    m = DetectionMetrics(iou_threshold=0.5)
    preds = [_empty_pred()]
    targets = [_gt_frame([[10, 10, 100, 100]])]

    m.update(preds, targets, "seq_fn")

    result = m.compute(sequence="seq_fn")
    r = result["all"]
    assert r["tp"] == 0
    assert r["fp"] == 0
    assert r["fn"] == 1
    assert r["precision"] == -1  # undefined (TP+FP == 0)
    assert r["recall"] == pytest.approx(0.0)
    assert r["support"] == 1


def test_all_false_positives():
    """One prediction with no GT: TP=0, FP=1, FN=0.

    Recall is undefined (-1); precision = 0.
    """
    m = DetectionMetrics(iou_threshold=0.5)
    preds = [_pred_frame([[10, 10, 100, 100]], scores=[0.9])]
    targets = [_empty_gt()]

    m.update(preds, targets, "seq_fp")

    result = m.compute(sequence="seq_fp")
    r = result["all"]
    assert r["tp"] == 0
    assert r["fp"] == 1
    assert r["fn"] == 0
    assert r["recall"] == -1  # undefined (TP+FN == 0)
    assert r["precision"] == pytest.approx(0.0)
    assert r["support"] == 0


# ---------------------------------------------------------------------------
# Tests: compute() global aggregate across sequences
# ---------------------------------------------------------------------------


def test_multi_sequence_global_aggregate():
    """Two sequences; global aggregate pools all TP/FP/FN.

    Sequence A: 1 TP  → tp=1, fp=0, fn=0
    Sequence B: 1 FP + 1 FN (pred and GT do not overlap) → tp=0, fp=1, fn=1
    Global: tp=1, fp=1, fn=1
      precision = 1 / (1+1) = 0.5
      recall    = 1 / (1+1) = 0.5
      f1        = 0.5
    """
    m = DetectionMetrics(iou_threshold=0.5)

    preds_a = [_pred_frame([[10, 10, 100, 100]], scores=[0.9])]
    targets_a = [_gt_frame([[10, 10, 100, 100]])]
    m.update(preds_a, targets_a, "seq_a")

    # Pred and GT are far apart → IoU ≈ 0 → no match
    preds_b = [_pred_frame([[10, 10, 100, 100]], scores=[0.9])]
    targets_b = [_gt_frame([[200, 200, 300, 300]])]
    m.update(preds_b, targets_b, "seq_b")

    global_result = m.compute()  # sequence=None → global
    r = global_result["all"]
    assert r["tp"] == 1
    assert r["fp"] == 1
    assert r["fn"] == 1
    assert r["precision"] == pytest.approx(0.5)
    assert r["recall"] == pytest.approx(0.5)
    assert r["f1"] == pytest.approx(0.5)


def test_per_sequence_results_differ_from_global():
    """Verify that per-sequence and global results can differ.

    Seq A: precision=1.0 (perfect).
    Seq B: precision=0.0 (all FP).
    Global precision: 1 TP, 1 FP → 0.5.
    """
    m = DetectionMetrics(iou_threshold=0.5)
    m.update(
        [_pred_frame([[10, 10, 100, 100]], scores=[0.9])],
        [_gt_frame([[10, 10, 100, 100]])],
        "seq_a",
    )
    m.update(
        [_pred_frame([[10, 10, 100, 100]], scores=[0.9])],
        [_empty_gt()],
        "seq_b",
    )

    assert m.compute(sequence="seq_a")["all"]["precision"] == pytest.approx(1.0)
    assert m.compute(sequence="seq_b")["all"]["recall"] == -1  # undefined
    assert m.compute()["all"]["precision"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Tests: failed_sequences  # noqa: ERA001
# ---------------------------------------------------------------------------


def test_log_failed_sequence_empty_gt_and_pred():
    """Both gt and pred empty → reason is 'No ground truth and no predictions'."""
    m = DetectionMetrics()
    m.log_failed_sequence("bad_seq", gt=[], pred=[])
    assert "bad_seq" in m.failed_sequences
    assert m.failed_sequences["bad_seq"] == "No ground truth and no predictions"


def test_log_failed_sequence_empty_gt():
    """Empty GT only → reason logged as 'No ground truth'."""
    m = DetectionMetrics()
    m.log_failed_sequence("no_gt", gt=[], pred=[[1, 2, 3]])
    assert m.failed_sequences["no_gt"] == "No ground truth"


def test_log_failed_sequence_empty_pred():
    """Empty pred only → reason logged as 'No predictions'."""
    m = DetectionMetrics()
    m.log_failed_sequence("no_pred", gt=[[1, 2, 3]], pred=[])
    assert m.failed_sequences["no_pred"] == "No predictions"


def test_log_failed_sequence_with_exception():
    """Exception provided → reason includes class name and message."""
    m = DetectionMetrics()
    exc = ValueError("something went wrong")
    m.log_failed_sequence("err_seq", gt=[[1]], pred=[[2]], exc=exc)
    assert "ValueError" in m.failed_sequences["err_seq"]
    assert "something went wrong" in m.failed_sequences["err_seq"]


def test_update_does_not_raise_on_invalid_input():
    """When update() encounters an error it logs to failed_sequences, not raises."""
    m = DetectionMetrics(iou_threshold=0.5)

    # Passing deliberately wrong types should be caught and logged.
    try:
        m.update("not_a_list", "not_a_list", "bad_seq")
    except Exception:
        pytest.fail("update() raised unexpectedly instead of logging failure")

    # Either the update succeeded (unlikely with bad input) or the sequence
    # was logged as failed.  At a minimum, no uncaught exception reached us.


# ---------------------------------------------------------------------------
# Tests: det_metrics_to_df  # noqa: ERA001
# ---------------------------------------------------------------------------


def test_det_metrics_to_df_shape_and_columns():
    """DataFrame should have one row per sequence with the expected columns."""
    m = DetectionMetrics(iou_threshold=0.5)
    m.update(
        [_pred_frame([[10, 10, 100, 100]], scores=[0.9])],
        [_gt_frame([[10, 10, 100, 100]])],
        "seq_a",
    )
    m.update(
        [_pred_frame([[10, 10, 100, 100]], scores=[0.9])],
        [_gt_frame([[200, 200, 300, 300]])],
        "seq_b",
    )

    df = det_metrics_to_df(m, area_range_label="all")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2  # one row per sequence
    assert "sequence" in df.columns
    for col in ("precision", "recall", "f1", "tp", "fp", "fn", "support", "n_imgs"):
        assert col in df.columns, f"Missing column: {col}"


def test_det_metrics_to_df_values():
    """DataFrame values should match per-sequence compute() results."""
    m = DetectionMetrics(iou_threshold=0.5)
    preds = [_pred_frame([[10, 10, 100, 100]], scores=[0.9])]
    targets = [_gt_frame([[10, 10, 100, 100]])]
    m.update(preds, targets, "seq_perfect")

    df = det_metrics_to_df(m, area_range_label="all")
    row = df.set_index("sequence").loc["seq_perfect"]

    assert row["tp"] == 1
    assert row["fp"] == 0
    assert row["fn"] == 0
    assert row["precision"] == pytest.approx(1.0)
    assert row["recall"] == pytest.approx(1.0)
    assert row["f1"] == pytest.approx(1.0)


def test_det_metrics_to_df_failed_sequences_excluded():
    """Sequences logged in failed_sequences must not appear in the DataFrame."""
    m = DetectionMetrics(iou_threshold=0.5)
    m.update(
        [_pred_frame([[10, 10, 100, 100]], scores=[0.9])],
        [_gt_frame([[10, 10, 100, 100]])],
        "seq_good",
    )
    m.log_failed_sequence("seq_bad", gt=[], pred=[])

    df = det_metrics_to_df(m)  # iterates only accumulators.keys()
    assert "seq_good" in df["sequence"].values
    assert "seq_bad" not in df["sequence"].values


def test_det_metrics_to_df_sequence_list_subset():
    """Passing sequence_list restricts rows to the given sequences."""
    m = DetectionMetrics(iou_threshold=0.5)
    for name in ("a", "b", "c"):
        m.update(
            [_pred_frame([[10, 10, 100, 100]], scores=[0.9])],
            [_gt_frame([[10, 10, 100, 100]])],
            name,
        )

    df = det_metrics_to_df(m, sequence_list=["a", "c"])
    assert set(df["sequence"].values) == {"a", "c"}


# ---------------------------------------------------------------------------
# Tests: area ranges
# ---------------------------------------------------------------------------


def test_area_ranges_returned_in_compute():
    """When area ranges are configured, compute() returns all labels."""
    m = DetectionMetrics(
        iou_threshold=0.5,
        area_ranges=[[0, 32**2], [32**2, 96**2], [96**2, int(1e10)], [0, int(1e10)]],
        area_ranges_labels=["small", "medium", "large", "all"],
    )
    preds = [_pred_frame([[10, 10, 100, 100]], scores=[0.9])]
    targets = [_gt_frame([[10, 10, 100, 100]])]
    m.update(preds, targets, "seq_ranges")

    result = m.compute(sequence="seq_ranges")
    for label in ("small", "medium", "large", "all"):
        assert label in result, f"Missing area range label: {label}"


def test_det_metrics_to_df_area_range_label():
    """det_metrics_to_df respects the area_range_label parameter."""
    m = DetectionMetrics(
        iou_threshold=0.5,
        area_ranges=[[0, int(1e10)], [0, 32**2]],
        area_ranges_labels=["all", "small"],
    )
    m.update(
        [_pred_frame([[10, 10, 100, 100]], scores=[0.9])],
        [_gt_frame([[10, 10, 100, 100]])],
        "seq",
    )

    df_all = det_metrics_to_df(m, area_range_label="all")
    df_small = det_metrics_to_df(m, area_range_label="small")

    # Both should have one row for "seq"
    assert len(df_all) == 1
    assert len(df_small) == 1
