"""HTML comparison report builder for tracking metrics."""

import html
import json
import pathlib

import pandas as pd

from .utils import COUNT_METRICS, OVERALL_LABEL

#: Metrics aggregated by summation in the summary row (count metrics); all other
#: columns use the pooled OVERALL value. Sourced from a single shared constant.
_SUM_METRICS = set(COUNT_METRICS)

_DISPLAY_NAME = {"TrackingMetrics": "MOT Metrics", "HOTAMetrics": "HOTA Metrics"}

_MODEL_COLORS = [
    "rgba(233,69,96,0.8)",
    "rgba(52,168,235,0.8)",
    "rgba(52,211,153,0.8)",
    "rgba(251,191,36,0.8)",
    "rgba(167,139,250,0.8)",
    "rgba(251,146,60,0.8)",
]


def _h(s: str) -> str:
    """Escape *s* for use in an HTML attribute value or text node.

    Parameters
    ----------
    s : str
        Raw string to escape.

    Returns:
    -------
    str
        HTML-escaped string.
    """
    return html.escape(s)


def _js(s: str) -> str:
    """Encode *s* as a JS string literal safe for use inside an HTML attribute.

    Parameters
    ----------
    s : str
        Raw string to encode.

    Returns:
    -------
    str
        JSON-serialised and HTML-escaped string.
    """
    return html.escape(json.dumps(s))


def _overall_value(df: pd.DataFrame, col: str) -> float:
    """Return the pooled ``OVERALL`` row value for *col*, or NaN if absent.

    Parameters
    ----------
    df : pd.DataFrame
        Per-metric DataFrame with a ``sequence`` column, optionally containing
        a pooled row labelled :data:`OVERALL_LABEL`.
    col : str
        Metric column name.

    Returns:
    -------
    float
        The pooled value, or NaN when no ``OVERALL`` row is present.
    """
    overall = df.loc[df["sequence"] == OVERALL_LABEL, col]
    return float(overall.iloc[0]) if not overall.empty else float("nan")


def _agg(df: pd.DataFrame, col: str) -> float:
    """Aggregate a metric column across sequences for the summary row.

    Count-based metrics (e.g. ``num_switches``) are summed over the per-sequence
    rows. Every other (ratio/derived) metric uses the pooled, dataset-level
    ``OVERALL`` value, since a plain mean of per-sequence ratios is not the
    correct aggregate for metrics like ``hota`` and ``idf1``. When no pooled row
    is present (older callers), it falls back to the per-sequence mean.

    Parameters
    ----------
    df : pd.DataFrame
        Per-metric DataFrame with a ``sequence`` column, optionally containing
        a pooled row labelled :data:`OVERALL_LABEL`.
    col : str
        Metric column name; used to decide the aggregation strategy.

    Returns:
    -------
    float
        Sum for count-based metrics, pooled ``OVERALL`` value for ratio metrics.
    """
    per_seq = df.loc[df["sequence"] != OVERALL_LABEL, col]
    if col in _SUM_METRICS:
        return per_seq.sum()
    overall = _overall_value(df, col)
    return overall if pd.notna(overall) else per_seq.mean()


def _round_or_none(val: float) -> float | None:
    """Round *val* to two decimal places, or return ``None`` if it is NaN.

    Parameters
    ----------
    val : float
        Numeric value to round.

    Returns:
    -------
    float or None
        Rounded value, or ``None`` when *val* is NaN / NA.
    """
    return round(float(val), 2) if pd.notna(val) else None


def _cell_value(dfs: dict, pf: str, mn: str, col: str, seq: str) -> float:
    """Look up a single metric value from the nested data dict.

    Parameters
    ----------
    dfs : dict
        Nested data dict: ``{pred_field: {metric_name: df}}``.
    pf : str
        Prediction field name (model identifier).
    mn : str
        Metric group name (e.g. ``"TrackingMetrics"``).
    col : str
        Metric column name.
    seq : str
        Sequence name.

    Returns:
    -------
    float
        Metric value, or NaN if the sequence is absent from the DataFrame.
    """
    return dfs[pf][mn].set_index("sequence").reindex([seq]).iloc[0][col]


def _fmt_cell(val: float) -> str:
    """Format a metric value for display in a table cell.

    Parameters
    ----------
    val : float
        Metric value (may be NaN).

    Returns:
    -------
    str
        Empty string for NaN values; ``"{val:.2f}"`` otherwise.
    """
    return "" if pd.isna(val) else f"{val:.2f}"


def _build_diff_controls(pred_fields: list, show_diff: bool) -> str:
    """Build the diff model-selector HTML, or return empty string.

    Parameters
    ----------
    pred_fields : list
        Ordered list of prediction field names.
    show_diff : bool
        When ``False`` the function returns an empty string immediately.

    Returns:
    -------
    str
        HTML ``<div>`` containing two ``<select>`` elements, or ``""``.
    """
    if not show_diff:
        return ""
    opts_a = "".join(
        f'<option value="{_h(pf)}">{_h(pf)}</option>' for pf in pred_fields
    )
    opts_b = "".join(
        f'<option value="{_h(pf)}" {"selected" if i == 1 else ""}>{_h(pf)}</option>'
        for i, pf in enumerate(pred_fields)
    )
    return (
        '<div style="margin-bottom:10px;font-size:12px;">'
        f'Compare: <select id="diff-a" class="diff-sel"'
        f' onchange="updateDiff()">{opts_a}</select>'
        f'&nbsp;vs&nbsp;<select id="diff-b" class="diff-sel"'
        f' onchange="updateDiff()">{opts_b}</select>'
        "</div>"
    )


def _assemble_html_table(
    pred_fields: list,
    metric_names: list,
    metric_cols: dict,
    sequences: list,
    dfs: dict,
    *,
    show_diff: bool,
    pf_idx: dict,
    n_cols_per_model: int,
    pf_mn_col: list,
    mn_col: list,
) -> str:
    """Assemble the full ``<table>`` HTML string from pre-computed layout values.

    Parameters
    ----------
    pred_fields : list
        Ordered list of prediction field names.
    metric_names : list
        Ordered list of metric group names.
    metric_cols : dict
        Mapping from metric name to its column names.
    sequences : list
        Sorted list of all sequence names.
    dfs : dict
        Nested data dict: ``{pred_field: {metric_name: df}}``.
    show_diff : bool
        Whether to render diff columns and controls.
    pf_idx : dict
        Mapping from prediction field name to its 0-based index.
    n_cols_per_model : int
        Number of metric columns per model (used for header colspan).

    Returns:
    -------
    str
        Complete diff-controls + ``<table>`` HTML.
    """
    n_regular_cols = len(pf_mn_col)

    sortable_headers = "".join(
        f'<th class="sortable" data-label="{_h(col)}" data-pf="{pf_idx[pf]}"'
        f' onclick="sortTable({i + 1})" title="Sort by {_h(col)}">{_h(col)}</th>'
        for i, (pf, mn, col) in enumerate(pf_mn_col)
    )
    if show_diff:
        sortable_headers += "".join(
            f'<th class="sortable" data-label="Δ {_h(col)}"'
            f' onclick="sortTable({n_regular_cols + i + 1})"'
            f' title="Sort by Δ {_h(col)}">Δ {_h(col)}</th>'
            for i, (mn, col) in enumerate(mn_col)
        )

    table_rows = "".join(
        "<tr><td>"
        + _h(seq)
        + "</td>"
        + "".join(
            f'<td style="text-align:right;" data-pf="{pf_idx[pf]}">'
            f"{_fmt_cell(_cell_value(dfs, pf, mn, col, seq))}</td>"
            for pf, mn, col in pf_mn_col
        )
        + (
            "".join(
                f'<td style="text-align:right;" data-diff'
                f' data-mn="{_h(mn)}" data-col="{_h(col)}"'
                f' data-seq="{_h(seq)}"></td>'
                for mn, col in mn_col
            )
            if show_diff
            else ""
        )
        + "</tr>"
        for seq in sequences
    )

    agg_cells = "".join(
        f'<td style="text-align:right;" data-pf="{pf_idx[pf]}">'
        f"{_agg(dfs[pf][mn], col):.2f}</td>"
        for pf, mn, col in pf_mn_col
    )
    if show_diff:
        agg_cells += "".join(
            f'<td style="text-align:right;" data-diff-mean'
            f' data-mn="{_h(mn)}" data-col="{_h(col)}"></td>'
            for mn, col in mn_col
        )

    pred_field_headers = "".join(
        f'<th colspan="{n_cols_per_model}" data-pf="{pf_idx[pf]}"'
        f' style="border-left:2px solid #e94560;">{_h(pf)}</th>'
        for pf in pred_fields
    )
    if show_diff:
        pred_field_headers += (
            f'<th colspan="{n_cols_per_model}"'
            f' style="border-left:2px solid #e94560;">'
            f'<span id="diff-label">Δ</span></th>'
        )

    metric_name_headers = "".join(
        f'<th colspan="{len(metric_cols[mn])}" data-pf="{pf_idx[pf]}"'
        f' style="border-left:2px solid #0f3460;">'
        f"{_h(_DISPLAY_NAME.get(mn, mn))}</th>"
        for pf in pred_fields
        for mn in metric_names
    )
    if show_diff:
        metric_name_headers += "".join(
            f'<th colspan="{len(metric_cols[mn])}"'
            f' style="border-left:2px solid #0f3460;">'
            f"{_h(_DISPLAY_NAME.get(mn, mn))}</th>"
            for mn in metric_names
        )

    diff_controls = _build_diff_controls(pred_fields, show_diff)

    return (
        diff_controls
        + f"""<table id="seq-table">
    <thead>
      <tr><th rowspan="3">Sequence</th>{pred_field_headers}</tr>
      <tr>{metric_name_headers}</tr>
      <tr>{sortable_headers}</tr>
    </thead>
    <tbody>
      {table_rows}
      <tr id="mean-row" style="font-weight:bold;border-top:2px solid #e94560;">
        <td>OVERALL / SUM</td>{agg_cells}
      </tr>
    </tbody>
  </table>"""
    )


def _build_chart_section(
    pred_fields: list,
    metric_names: list,
    metric_cols: dict,
    dfs: dict,
    color_map: dict,
) -> tuple[dict, str, str, str]:
    """Build the chart-data dict, model checkboxes, tab buttons, and chart grids.

    Parameters
    ----------
    pred_fields : list
        Ordered list of prediction field names (one per model).
    metric_names : list
        Ordered list of metric group names.
    metric_cols : dict
        Mapping from metric name to its column names.
    dfs : dict
        Nested data dict: ``{pred_field: {metric_name: df}}``.
    color_map : dict
        Mapping from prediction field name to CSS colour string.

    Returns:
    -------
    tuple[dict, str, str, str]
        ``(chart_data, checkboxes_html, tab_buttons_html, chart_grids_html)``
    """
    chart_data: dict = {}
    for mn in metric_names:
        chart_data[mn] = {}
        for col in metric_cols[mn]:
            chart_data[mn][col] = {
                pf: _round_or_none(_agg(dfs[pf][mn], col)) for pf in pred_fields
            }

    checkboxes_html = "".join(
        f'<label style="margin-right:16px;cursor:pointer;color:{color_map[pf]};">'
        f'<input type="checkbox" checked'
        f' onchange="toggleModel({_js(pf)})" style="margin-right:4px;">'
        f"{_h(pf)}</label>"
        for pf in pred_fields
    )

    tab_buttons = "".join(
        f'<button onclick="showTab({_js(mn)})" id="tab-{_h(mn)}" '
        f'style="margin-right:8px;padding:6px 14px;cursor:pointer;'
        f"background:{'#e94560' if i == 0 else '#1a1a2e'};"
        f'color:#fff;border:1px solid #e94560;border-radius:4px;">'
        f"{_h(_DISPLAY_NAME.get(mn, mn))}</button>"
        for i, mn in enumerate(metric_names)
    )

    chart_grids = ""
    for mn in metric_names:
        cols = metric_cols[mn]
        canvases = "".join(
            f'<div class="chart-card"><canvas id="chart-{mn}-{col}"></canvas></div>'
            for col in cols
        )
        display = "grid" if mn == metric_names[0] else "none"
        chart_grids += (
            f'<div id="grid-{mn}" class="chart-grid" style="display:{display};">'
            f"{canvases}</div>"
        )

    return chart_data, checkboxes_html, tab_buttons, chart_grids


def _build_table_html(
    pred_fields: list,
    metric_names: list,
    metric_cols: dict,
    sequences: list,
    dfs: dict,
) -> tuple[dict, str]:
    """Build the per-sequence table data dict and HTML table string.

    Parameters
    ----------
    pred_fields : list
        Ordered list of prediction field names.
    metric_names : list
        Ordered list of metric group names.
    metric_cols : dict
        Mapping from metric name to its column names.
    sequences : list
        Sorted list of all sequence names.
    dfs : dict
        Nested data dict: ``{pred_field: {metric_name: df}}``.

    Returns:
    -------
    tuple[dict, str]
        ``(table_data, table_html)`` where *table_data* is used for JS diff
        computation and *table_html* is the rendered ``<table>`` HTML.
    """
    table_data = {
        pf: {
            mn: {
                col: {
                    seq: _round_or_none(_cell_value(dfs, pf, mn, col, seq))
                    for seq in sequences
                }
                for col in metric_cols[mn]
            }
            for mn in metric_names
        }
        for pf in pred_fields
    }
    show_diff = len(pred_fields) >= 2  # noqa: PLR2004
    pf_idx = {pf: i for i, pf in enumerate(pred_fields)}
    n_cols_per_model = sum(len(metric_cols[mn]) for mn in metric_names)
    pf_mn_col = [
        (pf, mn, col)
        for pf in pred_fields
        for mn in metric_names
        for col in metric_cols[mn]
    ]
    mn_col = [(mn, col) for mn in metric_names for col in metric_cols[mn]]
    table_html = _assemble_html_table(
        pred_fields,
        metric_names,
        metric_cols,
        sequences,
        dfs,
        show_diff=show_diff,
        pf_idx=pf_idx,
        n_cols_per_model=n_cols_per_model,
        pf_mn_col=pf_mn_col,
        mn_col=mn_col,
    )
    return table_data, table_html


def build_comparison_html(dfs: dict) -> str:
    """Build an interactive HTML report with bar charts and a comparison table.

    Parameters
    ----------
    dfs : dict
        Nested dict: ``{pred_field: {"TrackingMetrics": df, "HOTAMetrics": df}}``.
        Each DataFrame has a ``sequence`` column plus numeric metric columns.

    Returns:
    -------
    str
        Fully rendered HTML document as a string.

    Raises:
    ------
    ValueError
        If *dfs* is empty.
    """
    if not dfs:
        raise ValueError("dfs must contain at least one element")

    pred_fields = list(dfs.keys())
    metric_names = list(next(iter(dfs.values())).keys())
    # The pooled OVERALL row is rendered as the summary row, not as a per-sequence
    # row, so exclude it from the sequence list used for the body and charts.
    sequences = sorted(
        set.intersection(
            *[
                set(pf_val[mn_key]["sequence"])
                for pf_val in dfs.values()
                for mn_key in metric_names
            ]
        )
        - {OVERALL_LABEL}
    )
    metric_cols = {
        m: [c for c in next(iter(dfs.values()))[m].columns if c != "sequence"]
        for m in metric_names
    }
    color_map = {
        pf: _MODEL_COLORS[i % len(_MODEL_COLORS)] for i, pf in enumerate(pred_fields)
    }
    # Summary-row values keyed for client-side diff (Δ) computation. Uses the
    # same aggregation as the rendered summary row (_agg) so the mean fallback
    # for legacy inputs without an OVERALL row stays consistent.
    overall_data = {
        pf: {
            mn: {col: _round_or_none(_agg(dfs[pf][mn], col)) for col in metric_cols[mn]}
            for mn in metric_names
        }
        for pf in pred_fields
    }
    chart_data, checkboxes_html, tab_buttons, chart_grids = _build_chart_section(
        pred_fields, metric_names, metric_cols, dfs, color_map
    )
    table_data, table_html = _build_table_html(
        pred_fields, metric_names, metric_cols, sequences, dfs
    )
    template = (pathlib.Path(__file__).parent / "comparison_report.html").read_text(
        encoding="utf-8"
    )

    return (
        template.replace("__SUM_METRICS__", json.dumps(sorted(_SUM_METRICS)))
        .replace("__CHART_DATA__", json.dumps(chart_data))
        .replace("__OVERALL_DATA__", json.dumps(overall_data))
        .replace("__TABLE_DATA__", json.dumps(table_data))
        .replace("__METRIC_COLS__", json.dumps(metric_cols))
        .replace("__SEQUENCES__", json.dumps(sequences))
        .replace("__PRED_FIELDS__", json.dumps(pred_fields))
        .replace("__METRIC_NAMES__", json.dumps(metric_names))
        .replace("__COLOR_MAP__", json.dumps(color_map))
        .replace("__CHECKBOXES_HTML__", checkboxes_html)
        .replace("__TAB_BUTTONS__", tab_buttons)
        .replace("__CHART_GRIDS__", chart_grids)
        .replace("__TABLE_HTML__", table_html)
    )
