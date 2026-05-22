"""Tests for seametrics.tracking.report."""

import math

import pandas as pd
import pytest

from seametrics.tracking.report import (
    _agg,
    _build_diff_controls,
    _cell_value,
    _fmt_cell,
    _h,
    _js,
    _round_or_none,
    build_comparison_html,
)

# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------


def _mot_df(sequences=("seq-1", "seq-2")):
    """Return a minimal MOT metrics DataFrame."""
    n = len(sequences)
    return pd.DataFrame(
        {
            "sequence": list(sequences),
            "mota": [50.0 + i * 10 for i in range(n)],
            "motp": [80.0 - i * 10 for i in range(n)],
            "num_switches": list(range(1, n + 1)),
            "num_false_positives": list(range(3, n + 3)),
            "num_misses": list(range(5, n + 5)),
            "num_fragmentations": list(range(n)),
            "num_frames": [100 * (i + 1) for i in range(n)],
            "mostly_tracked": list(range(2, n + 2)),
            "partially_tracked": [1] * n,
            "mostly_lost": [0] * n,
        }
    )


def _hota_df(sequences=("seq-1", "seq-2")):
    """Return a minimal HOTA metrics DataFrame."""
    n = len(sequences)
    return pd.DataFrame(
        {
            "sequence": list(sequences),
            "hota": [55.0 + i * 10 for i in range(n)],
            "deta": [50.0 + i * 10 for i in range(n)],
            "assa": [60.0 + i * 10 for i in range(n)],
            "num_unique_objects": list(range(3, n + 3)),
        }
    )


def _two_model_dfs(sequences=("seq-1", "seq-2")):
    """Return a two-model nested dfs dict."""
    return {
        "model_a": {
            "TrackingMetrics": _mot_df(sequences),
            "HOTAMetrics": _hota_df(sequences),
        },
        "model_b": {
            "TrackingMetrics": _mot_df(sequences),
            "HOTAMetrics": _hota_df(sequences),
        },
    }


def _one_model_dfs(sequences=("seq-1",)):
    """Return a single-model nested dfs dict."""
    return {
        "model_a": {"TrackingMetrics": _mot_df(sequences)},
    }


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


class TestHtmlEscape:
    """Tests for _h and _js."""

    def test_h_escapes_angle_brackets(self):
        assert _h("<b>bold</b>") == "&lt;b&gt;bold&lt;/b&gt;"

    def test_h_passthrough_safe_string(self):
        assert _h("hello") == "hello"

    def test_h_escapes_ampersand(self):
        assert "&amp;" in _h("a & b")

    def test_js_produces_html_escaped_json(self):
        result = _js("hello")
        assert "&quot;" in result

    def test_js_escapes_quotes(self):
        result = _js('say "hi"')
        assert "&quot;" in result

    def test_js_escapes_html_in_json(self):
        result = _js("<script>")
        assert "<script>" not in result


class TestAgg:
    """Tests for _agg."""

    def test_sum_metrics_are_summed(self):
        s = pd.Series([1.0, 2.0, 3.0])
        assert _agg(s, "num_switches") == pytest.approx(6.0)

    def test_sum_metrics_num_false_positives(self):
        s = pd.Series([10.0, 20.0])
        assert _agg(s, "num_false_positives") == pytest.approx(30.0)

    def test_ratio_metrics_are_averaged(self):
        s = pd.Series([0.0, 1.0])
        assert _agg(s, "mota") == pytest.approx(0.5)

    def test_ratio_hota_averaged(self):
        s = pd.Series([60.0, 80.0])
        assert _agg(s, "hota") == pytest.approx(70.0)


class TestRoundOrNone:
    """Tests for _round_or_none."""

    def test_rounds_to_two_decimal_places(self):
        assert _round_or_none(3.14159) == pytest.approx(3.14)

    def test_nan_returns_none(self):
        assert _round_or_none(float("nan")) is None

    def test_zero_returns_zero(self):
        assert _round_or_none(0.0) == pytest.approx(0.0)

    def test_pd_na_returns_none(self):
        assert _round_or_none(float("nan")) is None


class TestFmtCell:
    """Tests for _fmt_cell."""

    def test_nan_gives_empty_string(self):
        assert not _fmt_cell(float("nan"))

    def test_value_formatted_to_two_decimals(self):
        assert _fmt_cell(1.5) == "1.50"

    def test_integer_value(self):
        assert _fmt_cell(42.0) == "42.00"


class TestCellValue:
    """Tests for _cell_value."""

    def test_returns_correct_value(self):
        df = pd.DataFrame({"sequence": ["s1", "s2"], "mota": [0.5, 0.7]})
        dfs = {"pred_a": {"TrackingMetrics": df}}
        val = _cell_value(dfs, "pred_a", "TrackingMetrics", "mota", "s1")
        assert val == pytest.approx(0.5)

    def test_missing_sequence_returns_nan(self):
        df = pd.DataFrame({"sequence": ["s1"], "mota": [0.5]})
        dfs = {"pred_a": {"TrackingMetrics": df}}
        val = _cell_value(dfs, "pred_a", "TrackingMetrics", "mota", "s_missing")
        assert math.isnan(val)


class TestBuildDiffControls:
    """Tests for _build_diff_controls."""

    def test_disabled_returns_empty_string(self):
        assert not _build_diff_controls(["a", "b"], False)

    def test_enabled_contains_pred_fields(self):
        html = _build_diff_controls(["model_a", "model_b"], True)
        assert "model_a" in html
        assert "model_b" in html

    def test_enabled_contains_select_elements(self):
        html = _build_diff_controls(["model_a", "model_b"], True)
        assert html.count("<select") == 2

    def test_second_option_selected(self):
        html = _build_diff_controls(["model_a", "model_b"], True)
        assert "selected" in html


# ---------------------------------------------------------------------------
# Integration tests for build_comparison_html
# ---------------------------------------------------------------------------


class TestBuildComparisonHtml:
    """Tests for build_comparison_html."""

    def test_empty_dfs_raises_value_error(self):
        with pytest.raises(ValueError, match="at least one"):
            build_comparison_html({})

    def test_returns_string(self):
        result = build_comparison_html(_two_model_dfs())
        assert isinstance(result, str)

    def test_html_contains_pred_field_names(self):
        result = build_comparison_html(_two_model_dfs())
        assert "model_a" in result
        assert "model_b" in result

    def test_html_contains_sequence_names(self):
        result = build_comparison_html(_two_model_dfs())
        assert "seq-1" in result
        assert "seq-2" in result

    def test_html_contains_metric_column_names(self):
        result = build_comparison_html(_two_model_dfs())
        assert "mota" in result
        assert "hota" in result

    def test_two_model_report_has_diff_controls(self):
        result = build_comparison_html(_two_model_dfs())
        assert "diff-a" in result

    def test_single_model_no_diff_select(self):
        """With one model, the diff <select> widgets must not be rendered."""
        result = build_comparison_html(_one_model_dfs())
        assert '<select id="diff-a"' not in result

    def test_html_contains_table_element(self):
        result = build_comparison_html(_two_model_dfs())
        assert "<table" in result

    def test_html_contains_chart_data(self):
        result = build_comparison_html(_two_model_dfs())
        assert "chartData" in result or "__CHART_DATA__" not in result

    def test_mean_sum_row_present(self):
        result = build_comparison_html(_two_model_dfs())
        assert "MEAN / SUM" in result

    def test_single_sequence(self):
        dfs = _one_model_dfs(sequences=("only-seq",))
        result = build_comparison_html(dfs)
        assert "only-seq" in result

    def test_sum_metrics_aggregated_correctly(self):
        """num_switches is a SUM metric: value in HTML must equal sum not mean."""
        dfs = {
            "model_a": {
                "TrackingMetrics": pd.DataFrame(
                    {"sequence": ["s1", "s2"], "num_switches": [3.0, 7.0]}
                )
            }
        }
        result = build_comparison_html(dfs)
        assert "10.00" in result

    def test_color_map_cycles_for_many_models(self):
        """More models than palette entries must not raise."""
        many_models = {
            f"model_{i}": {"M": pd.DataFrame({"sequence": ["s1"], "v": [1.0]})}
            for i in range(8)
        }
        result = build_comparison_html(many_models)
        assert "model_0" in result
        assert "model_7" in result
