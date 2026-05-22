"""Tests for seametrics.tracking._box_utils."""

import numpy as np
import pytest

from seametrics.tracking._box_utils import (
    _box_cxcywh_to_xyxy,
    _box_xywh_to_xyxy,
    _box_xyxy_to_cxcywh,
    _box_xyxy_to_xywh,
    box_convert,
    box_denormalize,
)


class TestBoxDenormalize:
    """Tests for box_denormalize."""

    def test_empty_array_returned_unchanged(self):
        """Empty array should be returned without modification."""
        boxes = np.array([])
        result = box_denormalize(boxes, 640, 480)
        assert result.size == 0

    def test_pixel_coordinates_returned_unchanged(self):
        """Boxes with values > 1 are already in pixel space and must not be scaled."""
        boxes = np.array([100.0, 200.0, 300.0, 400.0])
        result = box_denormalize(boxes.copy(), 640, 480)
        np.testing.assert_array_equal(result, boxes)

    def test_normalized_boxes_scaled_correctly(self):
        """[x, y, w, h] layout: even indices scaled by img_w, odd by img_h."""
        boxes = np.array([0.1, 0.2, 0.3, 0.4])
        result = box_denormalize(boxes.copy(), 100, 50)
        expected = np.array([10.0, 10.0, 30.0, 20.0])
        np.testing.assert_allclose(result, expected)

    def test_exactly_one_boundary_normalized(self):
        """A box where all values are exactly 1.0 is still treated as normalized."""
        boxes = np.array([0.5, 0.5, 1.0, 1.0])
        result = box_denormalize(boxes.copy(), 200, 100)
        expected = np.array([100.0, 50.0, 200.0, 100.0])
        np.testing.assert_allclose(result, expected)


class TestBoxConvert:
    """Tests for box_convert and its private helpers."""

    def test_empty_array_returned_unchanged(self):
        """Empty input must pass straight through without errors."""
        boxes = np.empty((0, 4))
        result = box_convert(boxes, "xywh", "xyxy")
        assert result.size == 0

    def test_invalid_in_fmt_raises(self):
        boxes = np.array([[0.0, 0.0, 1.0, 1.0]])
        with pytest.raises(ValueError, match="Unsupported"):
            box_convert(boxes, "bad_fmt", "xyxy")

    def test_invalid_out_fmt_raises(self):
        boxes = np.array([[0.0, 0.0, 1.0, 1.0]])
        with pytest.raises(ValueError, match="Unsupported"):
            box_convert(boxes, "xyxy", "bad_fmt")

    def test_same_format_returns_copy(self):
        """Same in/out format: values unchanged, object is a copy not same array."""
        boxes = np.array([[1.0, 2.0, 3.0, 4.0]])
        result = box_convert(boxes, "xyxy", "xyxy")
        np.testing.assert_array_equal(result, boxes)
        assert result is not boxes

    def test_xywh_to_xyxy(self):
        """x1,y1 + w,h → x1,y1,x2,y2."""
        boxes = np.array([[1.0, 2.0, 3.0, 4.0]])
        result = box_convert(boxes, "xywh", "xyxy")
        np.testing.assert_allclose(result, [[1.0, 2.0, 4.0, 6.0]])

    def test_xyxy_to_xywh(self):
        """x1,y1,x2,y2 → x1,y1 + w,h."""
        boxes = np.array([[1.0, 2.0, 4.0, 6.0]])
        result = box_convert(boxes, "xyxy", "xywh")
        np.testing.assert_allclose(result, [[1.0, 2.0, 3.0, 4.0]])

    def test_cxcywh_to_xyxy(self):
        """Centre + w,h → corner coords."""
        boxes = np.array([[2.0, 3.0, 4.0, 6.0]])
        result = box_convert(boxes, "cxcywh", "xyxy")
        np.testing.assert_allclose(result, [[0.0, 0.0, 4.0, 6.0]])

    def test_xyxy_to_cxcywh(self):
        """Corner coords → centre + w,h."""
        boxes = np.array([[0.0, 0.0, 4.0, 6.0]])
        result = box_convert(boxes, "xyxy", "cxcywh")
        np.testing.assert_allclose(result, [[2.0, 3.0, 4.0, 6.0]])

    def test_xywh_to_cxcywh_via_intermediate(self):
        """Xywh → cxcywh: passes through intermediate xyxy conversion."""
        boxes = np.array([[1.0, 2.0, 4.0, 6.0]])
        result = box_convert(boxes, "xywh", "cxcywh")
        np.testing.assert_allclose(result, [[3.0, 5.0, 4.0, 6.0]])

    def test_cxcywh_to_xywh_via_intermediate(self):
        """Cxcywh → xywh: passes through intermediate xyxy conversion."""
        boxes = np.array([[3.0, 5.0, 4.0, 6.0]])
        result = box_convert(boxes, "cxcywh", "xywh")
        np.testing.assert_allclose(result, [[1.0, 2.0, 4.0, 6.0]])

    def test_multiple_boxes(self):
        """Conversion must handle batches of boxes correctly."""
        boxes = np.array([[0.0, 0.0, 2.0, 4.0], [1.0, 1.0, 3.0, 5.0]])
        result = box_convert(boxes, "xywh", "xyxy")
        np.testing.assert_allclose(result, [[0.0, 0.0, 2.0, 4.0], [1.0, 1.0, 4.0, 6.0]])


class TestPrivateConversions:
    """Tests for private single-step conversion helpers."""

    def test_box_xywh_to_xyxy(self):
        boxes = np.array([[1.0, 2.0, 3.0, 4.0]])
        result = _box_xywh_to_xyxy(boxes)
        np.testing.assert_allclose(result, [[1.0, 2.0, 4.0, 6.0]])

    def test_box_cxcywh_to_xyxy(self):
        boxes = np.array([[5.0, 6.0, 4.0, 2.0]])
        result = _box_cxcywh_to_xyxy(boxes)
        np.testing.assert_allclose(result, [[3.0, 5.0, 7.0, 7.0]])

    def test_box_xyxy_to_xywh(self):
        boxes = np.array([[1.0, 2.0, 4.0, 6.0]])
        result = _box_xyxy_to_xywh(boxes)
        np.testing.assert_allclose(result, [[1.0, 2.0, 3.0, 4.0]])

    def test_box_xyxy_to_cxcywh(self):
        boxes = np.array([[0.0, 0.0, 4.0, 6.0]])
        result = _box_xyxy_to_cxcywh(boxes)
        np.testing.assert_allclose(result, [[2.0, 3.0, 4.0, 6.0]])
