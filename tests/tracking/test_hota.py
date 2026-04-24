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


def _array(*rows):
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
    """
    1 GT track, 2 predicted IDs (a clean ID switch halfway through).

    Frame 1: GT id=1  matched to pred id=1  → TP
    Frame 2: GT id=1  matched to pred id=2  → TP (ID switch)

    Boxes are identical so IoU=1 and every threshold yields a TP.
    Expected (at each α):
        DetA = 2/2 = 1.0
        For each TP: tpa=1, gt_count=2, pr_count=1 → A = 1/(2+1-1) = 0.5
        AssA = 0.5
        HOTA = sqrt(1.0 * 0.5) ≈ 0.7071
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
    """
    1 GT track over 3 frames, only matched in 2 (1 FN, 0 FP).
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
    """Global compute() averages per-sequence results."""

    def setup_method(self):
        # seq1: perfect → HOTA=1
        gt1 = _array(_det(1, 1, 0, 0, 10, 10))
        pred1 = gt1.copy()

        # seq2: no predictions → HOTA=0
        gt2 = _array(_det(1, 1, 0, 0, 10, 10))
        pred2 = np.empty((0, 7))

        self.m = HOTAMetrics()
        self.m.update(gt1, pred1, "seq1")
        self.m.update(gt2, pred2, "seq2")

    def test_global_hota_is_mean(self):
        r = self.m.compute()  # sequence=None
        assert r["hota"] == pytest.approx(0.5, abs=1e-6)

    def test_unknown_sequence_raises(self):
        with pytest.raises(KeyError):
            self.m.compute("nonexistent")
