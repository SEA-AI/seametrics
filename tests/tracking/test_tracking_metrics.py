import numpy as np
import pytest

mm = pytest.importorskip("motmetrics", reason="motmetrics not installed")

from seametrics.tracking.track import TrackingMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _det(frame, obj_id, x1, y1, x2, y2):
    """Single MOT-format row: [frame, id, x1, y1, x2, y2]."""
    return [frame, obj_id, x1, y1, x2, y2]


def _array(*rows):
    return np.array(rows, dtype=float)


def _scalar(result, metric):
    """Extract the single scalar value from a per-sequence compute() result."""
    return list(result[metric].values())[0]


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

class TestNoPredicitions:
    """GT present, pred empty → everything missed."""

    def setup_method(self):
        gt = _array(
            _det(1, 1, 0, 0, 10, 10),
            _det(2, 1, 0, 0, 10, 10),
            _det(3, 1, 0, 0, 10, 10),
        )
        pred = _array(_det(1, 1, 100, 100, 110, 110))   # outside IoU threshold
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
# ID switch
# ---------------------------------------------------------------------------

class TestIDSwitch:
    """Same GT track matched to two different predicted IDs."""

    def setup_method(self):
        # Frames 1–2: GT id=1 → pred id=1  (correct)
        # Frame 3:    GT id=1 → pred id=2  (ID switch)
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


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:

    def test_unknown_sequence_raises(self):
        m = TrackingMetrics()
        with pytest.raises(Exception, match="Unknown sequence"):
            m.compute("nonexistent")

    def test_empty_metrics_global_returns_dict(self):
        # compute() with no sequences loaded should still return a dict
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
