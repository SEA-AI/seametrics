import math

import numpy as np
import pytest

from seametrics.tracking.track import TrackingMetrics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _det(frame, obj_id, x1, y1, x2, y2):
    """Single MOT-format row: [frame, id, x1, y1, x2, y2]."""
    return [frame, obj_id, x1, y1, x2, y2]


def _array(*rows: list) -> np.ndarray:
    return np.array(rows, dtype=float)


def _scalar(result, metric):
    """Extract the single scalar value from a per-sequence compute() result."""
    return next(iter(result[metric].values()))


# ---------------------------------------------------------------------------
# Perfect tracking
# ---------------------------------------------------------------------------


class TestPerfectTracking:
    """GT == pred in every frame."""

    def setup_method(self):
        gt = _array(
            _det(1, 1, 0, 0, 10, 10),
            _det(2, 1, 1, 1, 11, 11),
            _det(3, 1, 2, 2, 12, 12),
        )
        self.m = TrackingMetrics()
        self.m.update(gt, gt.copy(), "seq")

    def test_mota_is_one(self):
        r = self.m.compute("seq")
        assert _scalar(r, "mota") == pytest.approx(1.0)

    def test_no_misses(self):
        r = self.m.compute("seq")
        assert _scalar(r, "num_misses") == 0

    def test_no_false_positives(self):
        r = self.m.compute("seq")
        assert _scalar(r, "num_false_positives") == 0

    def test_no_id_switches(self):
        r = self.m.compute("seq")
        assert _scalar(r, "num_switches") == 0

    def test_num_frames(self):
        r = self.m.compute("seq")
        assert _scalar(r, "num_frames") == 3


# ---------------------------------------------------------------------------
# No predictions
# ---------------------------------------------------------------------------


class TestNoPredictions:
    """GT present, pred empty → everything missed."""

    def setup_method(self):
        gt = _array(
            _det(1, 1, 0, 0, 10, 10),
            _det(2, 1, 0, 0, 10, 10),
            _det(3, 1, 0, 0, 10, 10),
        )
        pred = _array(_det(1, 1, 100, 100, 110, 110))  # outside IoU threshold
        self.m = TrackingMetrics(max_iou=0.5)
        self.m.update(gt, pred, "seq")

    def test_mota(self):
        # FN=3 (gt obj missed in all frames), FP=1, IDSW=0, GT=3
        # MOTA = 1 - (3+1+0)/3 = -1/3
        r = self.m.compute("seq")
        assert _scalar(r, "mota") == pytest.approx(1 - 4 / 3)

    def test_misses(self):
        r = self.m.compute("seq")
        assert _scalar(r, "num_misses") == 3

    def test_false_positives(self):
        r = self.m.compute("seq")
        assert _scalar(r, "num_false_positives") == 1


# ---------------------------------------------------------------------------
# No ground truth
# ---------------------------------------------------------------------------


class TestTrackingNoGT:
    """All predictions, no ground truth — motmetrics counts every pred as FP.

    With zero GT objects, recall is undefined (NaN) and MOTA divides by zero
    (-inf). Precision is well-defined: TP / (TP + FP) = 0 / N = 0.
    """

    def setup_method(self):
        gt = np.empty((0, 6))
        pred = _array(
            _det(1, 1, 0, 0, 10, 10),
            _det(2, 1, 1, 1, 11, 11),
        )
        self.m = TrackingMetrics()
        self.m.update(gt, pred, "seq")

    def test_false_positives(self):
        r = self.m.compute("seq")
        assert _scalar(r, "num_false_positives") == 2

    def test_no_gt_objects(self):
        r = self.m.compute("seq")
        assert _scalar(r, "num_unique_objects") == 0

    def test_no_misses(self):
        r = self.m.compute("seq")
        assert _scalar(r, "num_misses") == 0

    def test_precision_is_zero(self):
        # precision = TP / (TP + FP) = 0 / 2
        r = self.m.compute("seq")
        assert _scalar(r, "precision") == pytest.approx(0.0)

    def test_recall_is_nan(self):
        # recall = TP / num_objects = 0 / 0 → undefined
        r = self.m.compute("seq")
        assert math.isnan(_scalar(r, "recall"))

    def test_mota_is_negative_inf(self):
        # MOTA = 1 - FP / num_objects = 1 - 2/0
        r = self.m.compute("seq")
        assert _scalar(r, "mota") == float("-inf")


class TestTrackingNoGTOverallPooling:
    """Pool a perfect sequence with an empty-GT sequence.

    motmetrics OVERALL recomputes ratio metrics from summed counts, so B's
    per-sequence NaN recall does not poison the dataset-level score.
    """

    def setup_method(self):
        gt_a = _array(_det(1, 1, 0, 0, 10, 10), _det(2, 1, 1, 1, 11, 11))
        pred_a = gt_a.copy()
        gt_b = np.empty((0, 6))
        pred_b = _array(_det(1, 1, 0, 0, 10, 10), _det(2, 1, 1, 1, 11, 11))

        self.m = TrackingMetrics()
        self.m.update(gt_a, pred_a, "A")
        self.m.update(gt_b, pred_b, "B")
        self.r = self.m.compute(["A", "B"])

    def test_empty_gt_sequence_recall_is_nan(self):
        assert math.isnan(self.r["recall"]["B"])

    def test_overall_precision(self):
        # pooled: TP=2, FP=2 → 2 / (2 + 2)
        assert self.r["precision"]["OVERALL"] == pytest.approx(0.5)

    def test_overall_recall(self):
        # pooled: TP=2, num_objects=2 → not NaN despite B's undefined recall
        assert self.r["recall"]["OVERALL"] == pytest.approx(1.0)

    def test_overall_mota(self):
        # Two pooled false positives against two GT object appearances.
        assert self.r["mota"]["OVERALL"] == pytest.approx(0.0)

    def test_overall_false_positives_summed(self):
        assert self.r["num_false_positives"]["OVERALL"] == 2

    def test_overall_mota_differs_from_per_sequence_mean(self):
        # Naive mean of per-sequence MOTA (1.0, -inf) is not the MOT-standard pool.
        per_seq_mean = np.mean([self.r["mota"]["A"], self.r["mota"]["B"]])
        assert self.r["mota"]["OVERALL"] != pytest.approx(per_seq_mean)


# ---------------------------------------------------------------------------
# ID switch
# ---------------------------------------------------------------------------


class TestIDSwitch:
    """Same GT track matched to two different predicted IDs."""

    def setup_method(self):
        # Frames 1-2: GT id=1 -> pred id=1  (correct)
        # Frame 3:    GT id=1 -> pred id=2  (ID switch)
        gt = _array(
            _det(1, 1, 0, 0, 10, 10),
            _det(2, 1, 0, 0, 10, 10),
            _det(3, 1, 0, 0, 10, 10),
        )
        pred = _array(
            _det(1, 1, 0, 0, 10, 10),
            _det(2, 1, 0, 0, 10, 10),
            _det(3, 2, 0, 0, 10, 10),
        )
        self.m = TrackingMetrics()
        self.m.update(gt, pred, "seq")

    def test_num_switches(self):
        r = self.m.compute("seq")
        assert _scalar(r, "num_switches") == 1

    def test_mota_penalised(self):
        # MOTA = 1 - (FN=0 + FP=0 + IDSW=1) / GT=3
        r = self.m.compute("seq")
        assert _scalar(r, "mota") == pytest.approx(1 - 1 / 3)

    def test_idf1_below_one(self):
        r = self.m.compute("seq")
        assert _scalar(r, "idf1") < 1.0


# ---------------------------------------------------------------------------
# Global aggregation (sequence=None)
# ---------------------------------------------------------------------------


class TestGlobalCompute:
    def setup_method(self):
        gt = _array(_det(1, 1, 0, 0, 10, 10))
        self.m = TrackingMetrics()
        self.m.update(gt, gt.copy(), "seq1")
        self.m.update(gt, gt.copy(), "seq2")

    def test_returns_dict(self):
        r = self.m.compute()
        assert isinstance(r, dict)

    def test_overall_row_present(self):
        r = self.m.compute()
        assert "OVERALL" in r["mota"]

    def test_both_sequences_present(self):
        r = self.m.compute()
        assert "seq1" in r["mota"]
        assert "seq2" in r["mota"]

    def test_overall_mota_is_one(self):
        r = self.m.compute()
        assert r["mota"]["OVERALL"] == pytest.approx(1.0)


class TestSubsetPooling:
    """compute(list) pools IDF1 counts over exactly the named subset.

    Two sequences, identical boxes:
      A (perfect): gt id1 frames 1,2 ; pred id1 frames 1,2
          -> IDTP=2 IDFP=0 IDFN=0 -> IDF1 = 4/4 = 1.0
      B (one miss): gt id1 frames 1,2 ; pred id1 frame 1 only
          -> IDTP=1 IDFP=0 IDFN=1 -> IDF1 = 2/3 = 0.6666667

    Pooled over {A, B}: IDTP=3 IDFP=0 IDFN=1 -> IDF1 = 6/7 = 0.8571429.
    A naive mean of per-sequence IDF1 would be (1.0 + 2/3)/2 = 0.8333333.
    """

    def setup_method(self):
        gt_a = _array(_det(1, 1, 0, 0, 10, 10), _det(2, 1, 0, 0, 10, 10))
        pred_a = gt_a.copy()
        gt_b = _array(_det(1, 1, 0, 0, 10, 10), _det(2, 1, 0, 0, 10, 10))
        pred_b = _array(_det(1, 1, 0, 0, 10, 10))  # frame 2 missed

        self.m = TrackingMetrics()
        self.m.update(gt_a, pred_a, "A")
        self.m.update(gt_b, pred_b, "B")

    def test_per_sequence_idf1_unchanged(self):
        assert _scalar(self.m.compute("A"), "idf1") == pytest.approx(1.0)
        assert _scalar(self.m.compute("B"), "idf1") == pytest.approx(2 / 3)

    def test_subset_overall_is_pooled(self):
        overall = self.m.compute(["A", "B"])["idf1"]["OVERALL"]
        assert overall == pytest.approx(6 / 7)

    def test_subset_overall_differs_from_mean(self):
        overall = self.m.compute(["A", "B"])["idf1"]["OVERALL"]
        mean = (
            _scalar(self.m.compute("A"), "idf1") + _scalar(self.m.compute("B"), "idf1")
        ) / 2
        assert overall != pytest.approx(mean, abs=1e-3)

    def test_subset_matches_none_when_all_sequences_listed(self):
        assert self.m.compute(["A", "B"])["idf1"]["OVERALL"] == pytest.approx(
            self.m.compute()["idf1"]["OVERALL"]
        )

    def test_unknown_sequence_in_list_raises(self):
        with pytest.raises(ValueError, match="Unknown sequence"):
            self.m.compute(["A", "nope"])

    def test_duplicate_sequence_in_list_raises(self):
        # A duplicate name would pool the same accumulator twice and skew scores.
        with pytest.raises(ValueError, match="Duplicate sequence"):
            self.m.compute(["A", "A"])


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_unknown_sequence_raises(self):
        m = TrackingMetrics()
        with pytest.raises(Exception, match="Unknown sequence"):
            m.compute("nonexistent")

    def test_global_compute_returns_dict(self):
        # compute() with at least one sequence loaded should return a dict
        m = TrackingMetrics()
        gt = _array(_det(1, 1, 0, 0, 10, 10))
        m.update(gt, gt.copy(), "seq")
        r = m.compute()
        assert "mota" in r


# ---------------------------------------------------------------------------
# log_failed_sequence
# ---------------------------------------------------------------------------


class TestLogFailedSequence:
    def test_no_gt_and_no_pred(self):
        m = TrackingMetrics()
        m.log_failed_sequence("s", [], [])
        assert m.failed_sequences["s"] == "No ground truth and no predictions"

    def test_no_gt(self):
        m = TrackingMetrics()
        m.log_failed_sequence("s", [], [1])
        assert m.failed_sequences["s"] == "No ground truth"

    def test_no_pred(self):
        m = TrackingMetrics()
        m.log_failed_sequence("s", [1], [])
        assert m.failed_sequences["s"] == "No predictions"

    def test_missing_ids(self):
        m = TrackingMetrics()
        m.log_failed_sequence("s", [1], [1])
        assert m.failed_sequences["s"] == "Missing IDs from GT or Pred"
