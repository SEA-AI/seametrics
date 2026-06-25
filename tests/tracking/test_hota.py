import math

import numpy as np
import pytest

from seametrics.tracking.hota import HOTAMetrics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _det(frame, obj_id, x1, y1, x2, y2, conf=1.0):
    """Build a single row in MOT format: [frame, id, x1, y1, x2, y2, conf]."""
    return [frame, obj_id, x1, y1, x2, y2, conf]


def _array(*rows: list) -> np.ndarray:
    return np.array(rows, dtype=float)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHOTAPerfect:
    """Perfect tracker: pred == gt in every frame."""

    def setup_method(self):
        gt = _array(
            _det(1, 1, 0, 0, 10, 10),
            _det(2, 1, 1, 1, 11, 11),
            _det(3, 1, 2, 2, 12, 12),
        )
        pred = gt.copy()
        pred[:, 6] = 0.9  # confidence column doesn't affect HOTA

        self.m = HOTAMetrics()
        self.m.update(gt, pred, "seq")

    def test_hota_is_one(self):
        r = self.m.compute("seq")
        assert r["hota"] == pytest.approx(1.0, abs=1e-6)

    def test_deta_is_one(self):
        r = self.m.compute("seq")
        assert r["deta"] == pytest.approx(1.0, abs=1e-6)

    def test_assa_is_one(self):
        r = self.m.compute("seq")
        assert r["assa"] == pytest.approx(1.0, abs=1e-6)

    def test_loca_is_one(self):
        r = self.m.compute("seq")
        assert r["loca"] == pytest.approx(1.0, abs=1e-6)


class TestHOTANoGT:
    """All predictions, no ground truth → DetA = 0 → HOTA = 0."""

    def setup_method(self):
        gt = np.empty((0, 7))
        pred = _array(
            _det(1, 1, 0, 0, 10, 10, 0.9),
            _det(2, 1, 1, 1, 11, 11, 0.8),
        )
        self.m = HOTAMetrics()
        self.m.update(gt, pred, "seq")

    def test_hota_is_zero(self):
        r = self.m.compute("seq")
        assert r["hota"] == pytest.approx(0.0, abs=1e-6)

    def test_deta_is_zero(self):
        r = self.m.compute("seq")
        assert r["deta"] == pytest.approx(0.0, abs=1e-6)


class TestHOTANoPred:
    """All ground truth, no predictions → DetA = 0 → HOTA = 0."""

    def setup_method(self):
        gt = _array(
            _det(1, 1, 0, 0, 10, 10),
            _det(2, 1, 1, 1, 11, 11),
        )
        pred = np.empty((0, 7))
        self.m = HOTAMetrics()
        self.m.update(gt, pred, "seq")

    def test_hota_is_zero(self):
        r = self.m.compute("seq")
        assert r["hota"] == pytest.approx(0.0, abs=1e-6)

    def test_deta_is_zero(self):
        r = self.m.compute("seq")
        assert r["deta"] == pytest.approx(0.0, abs=1e-6)


class TestHOTAIDSwitch:
    """1 GT track, 2 predicted IDs (a clean ID switch halfway through).

    Frame 1: GT id=1  matched to pred id=1  → TP
    Frame 2: GT id=1  matched to pred id=2  → TP (ID switch)

    Boxes are identical so IoU=1 and every threshold yields a TP.
    Expected (at each alpha):
        DetA = 2/2 = 1.0
        For each TP: tpa=1, gt_count=2, pr_count=1 -> A = 1/(2+1-1) = 0.5
        AssA = 0.5
        HOTA = sqrt(1.0 * 0.5) ~= 0.7071
    """

    def setup_method(self):
        gt = _array(
            _det(1, 1, 0, 0, 10, 10),
            _det(2, 1, 0, 0, 10, 10),
        )
        pred = _array(
            _det(1, 1, 0, 0, 10, 10),  # correct ID
            _det(2, 2, 0, 0, 10, 10),  # switched ID
        )
        self.m = HOTAMetrics()
        self.m.update(gt, pred, "seq")

    def test_deta_is_one(self):
        r = self.m.compute("seq")
        assert r["deta"] == pytest.approx(1.0, abs=1e-6)

    def test_assa_is_half(self):
        r = self.m.compute("seq")
        assert r["assa"] == pytest.approx(0.5, abs=1e-6)

    def test_hota(self):
        r = self.m.compute("seq")
        assert r["hota"] == pytest.approx(math.sqrt(0.5), abs=1e-6)


class TestHOTAPartialDetection:
    """1 GT track over 3 frames, only matched in 2 (1 FN, 0 FP).

    Expected DetA = 2/3 at all thresholds (IoU=1).
    """

    def setup_method(self):
        gt = _array(
            _det(1, 1, 0, 0, 10, 10),
            _det(2, 1, 0, 0, 10, 10),
            _det(3, 1, 0, 0, 10, 10),
        )
        pred = _array(
            _det(1, 1, 0, 0, 10, 10),
            _det(2, 1, 0, 0, 10, 10),
            # frame 3: no prediction
        )
        self.m = HOTAMetrics()
        self.m.update(gt, pred, "seq")

    def test_deta(self):
        r = self.m.compute("seq")
        assert r["deta"] == pytest.approx(2 / 3, abs=1e-6)


class TestHOTAGlobalAggregation:
    """Global compute() pools TP/FP/FN counts across sequences (MOT standard)."""

    def setup_method(self):
        # seq1: perfect → 1 TP, 0 FP, 0 FN
        gt1 = _array(_det(1, 1, 0, 0, 10, 10))
        pred1 = gt1.copy()

        # seq2: no predictions → 0 TP, 0 FP, 1 FN
        gt2 = _array(_det(1, 1, 0, 0, 10, 10))
        pred2 = np.empty((0, 7))

        self.m = HOTAMetrics()
        self.m.update(gt1, pred1, "seq1")
        self.m.update(gt2, pred2, "seq2")

    def test_global_hota_pools_counts(self):
        # Pooled: TP=1, FP=0, FN=1 → DetA=0.5, AssA=1.0 → HOTA=sqrt(0.5)
        r = self.m.compute()  # sequence=None
        assert r["hota"] == pytest.approx(0.5**0.5, abs=1e-6)

    def test_unknown_sequence_raises(self):
        with pytest.raises(ValueError, match="Unknown sequence"):
            self.m.compute("nonexistent")


class TestHOTASubsetPooling:
    """compute(list) pools exactly the named subset (MOT standard).

    Two sequences, identical boxes so IoU=1 and matching is threshold-independent:
      A (perfect): gt id1 frames 1,2 ; pred id1 frames 1,2
          -> TP=2 FP=0 FN=0, DetA=1 ; pair(1,1) cnt2 -> AssA=1 ; HOTA=1
      B (one miss): gt id1 frames 1,2 ; pred id1 frame 1 only
          -> TP=1 FP=0 FN=1 DetA=0.5 ; pair(1,1) cnt1 gtf2 prf1 -> AssA=0.5 HOTA=0.5

    Pooled over {A, B} (ids/frames offset so they don't collide):
      TP=3 FP=0 FN=1 -> DetA=3/4=0.75
      pair(1,1) cnt2 ass=2/(2+2-2)=1 (x2) ; pair(3,3) cnt1 ass=1/(2+1-1)=0.5
      AssA = (1+1+0.5)/3 = 0.8333333 -> HOTA = sqrt(0.75*0.8333333) = sqrt(0.625)
    A plain mean of per-sequence HOTA would instead be (1.0+0.5)/2 = 0.75.
    """

    def setup_method(self):
        gt_a = _array(_det(1, 1, 0, 0, 10, 10), _det(2, 1, 0, 0, 10, 10))
        pred_a = gt_a.copy()
        gt_b = _array(_det(1, 1, 0, 0, 10, 10), _det(2, 1, 0, 0, 10, 10))
        pred_b = _array(_det(1, 1, 0, 0, 10, 10))  # frame 2 missed
        # An extra, larger sequence to prove the subset excludes it.
        gt_c = _array(_det(1, 7, 0, 0, 10, 10), _det(1, 8, 20, 20, 30, 30))
        pred_c = gt_c.copy()

        self.m = HOTAMetrics()
        self.m.update(gt_a, pred_a, "A")
        self.m.update(gt_b, pred_b, "B")
        self.m.update(gt_c, pred_c, "C")

    def test_per_sequence_unchanged(self):
        assert self.m.compute("A")["hota"] == pytest.approx(1.0, abs=1e-6)
        assert self.m.compute("B")["hota"] == pytest.approx(0.5, abs=1e-6)

    def test_subset_pooled_value(self):
        pooled = self.m.compute(["A", "B"])["hota"]
        assert pooled == pytest.approx(math.sqrt(0.625), abs=1e-6)

    def test_subset_differs_from_naive_mean(self):
        pooled = self.m.compute(["A", "B"])["hota"]
        mean = (self.m.compute("A")["hota"] + self.m.compute("B")["hota"]) / 2
        assert pooled != pytest.approx(mean, abs=1e-3)

    def test_subset_excludes_unnamed_sequence(self):
        # Pooling {A, B} must ignore C, so it differs from pooling all three.
        assert self.m.compute(["A", "B"])["hota"] != pytest.approx(
            self.m.compute()["hota"], abs=1e-6
        )

    def test_unknown_sequence_in_list_raises(self):
        with pytest.raises(ValueError, match="Unknown sequence"):
            self.m.compute(["A", "nonexistent"])

    def test_duplicate_sequence_in_list_raises(self):
        # A duplicate name would pool the same accumulator twice and skew scores.
        with pytest.raises(ValueError, match="Duplicate sequence"):
            self.m.compute(["A", "A"])

    def test_compute_many_duplicate_sequence_raises(self):
        with pytest.raises(ValueError, match="Duplicate sequence"):
            self.m.compute_many(["A", "A"])

    def test_compute_many_matches_single_calls(self):
        bundle = self.m.compute_many(["A", "B"])
        assert bundle["hota"]["A"] == pytest.approx(self.m.compute("A")["hota"])
        assert bundle["hota"]["B"] == pytest.approx(self.m.compute("B")["hota"])
        assert bundle["hota"]["OVERALL"] == pytest.approx(
            self.m.compute(["A", "B"])["hota"]
        )


class TestHOTAMeanVsPooledManyObjects:
    """One easy sequence plus a many-object failure mode.

    A plain mean of per-sequence HOTA can look tolerable while the pooled
    dataset score reflects most object-frame associations failing.
    """

    def setup_method(self):
        easy_gt = _array(
            _det(1, 1, 0, 0, 10, 10),
            _det(2, 1, 0, 0, 10, 10),
            _det(3, 1, 0, 0, 10, 10),
        )
        hard_gt = _array(
            *[
                _det(frame, obj_id, 0, 0, 10, 10)
                for obj_id in range(1, 6)
                for frame in range(1, 6)
            ]
        )
        hard_pred = _array(*[_det(1, obj_id, 0, 0, 10, 10) for obj_id in range(1, 6)])

        self.m = HOTAMetrics()
        self.m.update(easy_gt, easy_gt.copy(), "seq_easy")
        self.m.update(hard_gt, hard_pred, "seq_hard")

    def test_per_sequence_scores(self):
        assert self.m.compute("seq_easy")["hota"] == pytest.approx(1.0, abs=1e-6)
        assert self.m.compute("seq_hard")["hota"] == pytest.approx(0.2, abs=1e-6)

    def test_pooled_below_naive_mean(self):
        easy = self.m.compute("seq_easy")["hota"]
        hard = self.m.compute("seq_hard")["hota"]
        mean = (easy + hard) / 2
        pooled = self.m.compute(["seq_easy", "seq_hard"])["hota"]
        assert mean == pytest.approx(0.6, abs=1e-6)
        assert pooled == pytest.approx(0.377964473, abs=1e-6)
        assert pooled < mean
        assert mean - pooled > 0.15
