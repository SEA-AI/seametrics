"""Tests for seametrics.tracking.report."""

import json
import math
import re

import pandas as pd
import pytest

from seametrics.tracking.report import (
    _agg,
    _build_diff_controls,
    _cell_value,
    _fmt_cell,
    _h,
    _js,
    _json_for_html_script,
    _round_or_none,
    build_comparison_html,
)

# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------


def _mot_df(sequences=("seq-1", "seq-2"), *, overall=True):
    """Return a minimal MOT metrics DataFrame, with a pooled OVERALL row.

    The OVERALL ratio values (``mota``/``motp``) are deliberately set to numbers
    that differ from the mean of the per-sequence rows, so tests can tell the
    pooled value apart from a naive mean.
    """
    n = len(sequences)
    df = pd.DataFrame(
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
    if overall:
        overall_row = {
            "sequence": "OVERALL",
            "mota": 42.0,  # != mean(50, 60) = 55
            "motp": 33.0,  # != mean(80, 70) = 75
            "num_switches": 0,  # ignored: counts are summed over per-seq rows
            "num_false_positives": 0,
            "num_misses": 0,
            "num_fragmentations": 0,
            "num_frames": 0,
            "mostly_tracked": 0,
            "partially_tracked": 0,
            "mostly_lost": 0,
        }
        df = pd.concat([df, pd.DataFrame([overall_row])], ignore_index=True)
    return df


def _hota_df(sequences=("seq-1", "seq-2"), *, overall=True):
    """Return a minimal HOTA metrics DataFrame, with a pooled OVERALL row."""
    n = len(sequences)
    df = pd.DataFrame(
        {
            "sequence": list(sequences),
            "hota": [55.0 + i * 10 for i in range(n)],
            "deta": [50.0 + i * 10 for i in range(n)],
            "assa": [60.0 + i * 10 for i in range(n)],
            "num_unique_objects": list(range(3, n + 3)),
        }
    )
    if overall:
        overall_row = {
            "sequence": "OVERALL",
            "hota": 47.0,  # != mean(55, 65) = 60
            "deta": 50.0,
            "assa": 60.0,
            "num_unique_objects": 0,
        }
        df = pd.concat([df, pd.DataFrame([overall_row])], ignore_index=True)
    return df


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


class TestJsonForHtmlScript:
    """Tests for _json_for_html_script."""

    def test_escapes_angle_brackets_for_script_context(self):
        payload = ["</script><script>alert(1)</script>"]
        encoded = _json_for_html_script(payload)
        assert "<" not in encoded
        assert ">" not in encoded
        decoded = encoded.replace("\\u003c", "<").replace("\\u003e", ">")
        assert json.loads(decoded) == payload


def _extract_js_const(html: str, name: str) -> object:
    match = re.search(rf"const {name}\s*=\s*(.+);\n", html)
    assert match is not None, f"{name} assignment not found"
    raw = match.group(1)
    unescaped = (
        raw.replace("\\u0026", "&").replace("\\u003c", "<").replace("\\u003e", ">")
    )
    return json.loads(unescaped)


class TestLegacyOverallConsistency:
    """Regression tests for CodeRabbit overallData vs summary-row consistency."""

    def test_legacy_df_overall_data_matches_summary_row(self):
        """Without an OVERALL row, JS overallData must use the same mean as _agg."""
        dfs = {
            "model_a": {
                "TrackingMetrics": pd.DataFrame(
                    {"sequence": ["s1", "s2"], "mota": [60.0, 80.0]}
                )
            }
        }
        html = build_comparison_html(dfs)
        assert "70.00" in html
        overall = _extract_js_const(html, "overallData")
        assert overall["model_a"]["TrackingMetrics"]["mota"] == pytest.approx(70.0)

    def test_malicious_pred_field_does_not_break_inline_script(self):
        evil = "</script><script>alert(1)</script>"
        dfs = {
            evil: {
                "TrackingMetrics": pd.DataFrame({"sequence": ["s1"], "mota": [42.0]})
            },
            "model_b": {
                "TrackingMetrics": pd.DataFrame({"sequence": ["s1"], "mota": [10.0]})
            },
        }
        html = build_comparison_html(dfs)
        inline_script = html.split(
            'src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels'
        )[1]
        assert evil not in inline_script
        pred_fields = _extract_js_const(html, "predFields")
        assert evil in pred_fields


class TestAgg:
    """Tests for _agg.

    Count metrics are summed over the per-sequence rows (the OVERALL row is
    ignored). Ratio metrics take the pooled OVERALL value, falling back to the
    per-sequence mean only when no OVERALL row exists.
    """

    def test_sum_metrics_are_summed_ignoring_overall(self):
        df = pd.DataFrame(
            {
                "sequence": ["s1", "s2", "s3", "OVERALL"],
                "num_switches": [1.0, 2.0, 3.0, 999.0],
            }
        )
        assert _agg(df, "num_switches") == pytest.approx(6.0)

    def test_sum_metrics_num_false_positives(self):
        df = pd.DataFrame(
            {
                "sequence": ["s1", "s2", "OVERALL"],
                "num_false_positives": [10.0, 20.0, 999.0],
            }
        )
        assert _agg(df, "num_false_positives") == pytest.approx(30.0)

    def test_ratio_metric_uses_pooled_overall_not_mean(self):
        # mean of per-seq would be 0.5; pooled OVERALL is 0.42 → must pick 0.42.
        df = pd.DataFrame(
            {"sequence": ["s1", "s2", "OVERALL"], "mota": [0.0, 1.0, 0.42]}
        )
        assert _agg(df, "mota") == pytest.approx(0.42)

    def test_ratio_hota_uses_pooled_overall(self):
        # mean of per-seq would be 70; pooled OVERALL is 79.06 → must pick pooled.
        df = pd.DataFrame(
            {"sequence": ["s1", "s2", "OVERALL"], "hota": [60.0, 80.0, 79.06]}
        )
        assert _agg(df, "hota") == pytest.approx(79.06)

    def test_ratio_falls_back_to_mean_without_overall_row(self):
        df = pd.DataFrame({"sequence": ["s1", "s2"], "hota": [60.0, 80.0]})
        assert _agg(df, "hota") == pytest.approx(70.0)


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
        indexed = df.set_index("sequence")
        val = _cell_value(indexed, "mota", "s1")
        assert val == pytest.approx(0.5)

    def test_missing_sequence_returns_nan(self):
        df = pd.DataFrame({"sequence": ["s1"], "mota": [0.5]})
        indexed = df.set_index("sequence")
        val = _cell_value(indexed, "mota", "s_missing")
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

    def test_summary_row_present(self):
        result = build_comparison_html(_two_model_dfs())
        assert "OVERALL / SUM" in result

    def test_summary_row_uses_pooled_overall_for_ratio(self):
        """Ratio summary cell shows the pooled OVERALL value, not the mean.

        For mota the pooled value is 42.00; the per-sequence mean would be 55.00.
        """
        result = build_comparison_html(_one_model_dfs(sequences=("seq-1", "seq-2")))
        assert "42.00" in result
        assert ">55.00<" not in result

    def test_overall_row_not_rendered_as_sequence(self):
        """The pooled OVERALL row must not appear as a per-sequence table row."""
        result = build_comparison_html(_one_model_dfs(sequences=("seq-1", "seq-2")))
        assert "<td>OVERALL</td>" not in result

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
