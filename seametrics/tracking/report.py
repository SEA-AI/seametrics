"""HTML comparison report builder for tracking metrics."""

import html
import json
import pathlib
from itertools import chain, product

import pandas as pd

from .constants import COUNT_METRICS, DISPLAY_NAMES, MODEL_COLORS, OVERALL_LABEL

#: Metrics aggregated by summation in the summary row (count metrics); all other
#: columns use the pooled OVERALL value. Sourced from a single shared constant.
_SUM_METRICS = set(COUNT_METRICS)


def _h(s: str) -> str:
    """Escape *s* for use in an HTML attribute value or text node."""
    return html.escape(s)


def _js(s: str) -> str:
    """Encode *s* as a JS string literal safe for use inside an HTML attribute."""
    return html.escape(json.dumps(s))


def _json_for_html_script(value: object) -> str:
    """Serialize *value* as JSON with ``&``, ``<``, ``>`` escaped for scripts."""
    return (
        json.dumps(value)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def _overall_value(df: pd.DataFrame, col: str) -> float:
    """Return the pooled ``OVERALL`` row value for *col*, or NaN if absent."""
    overall = df.loc[df["sequence"] == OVERALL_LABEL, col]
    return float(overall.iloc[0]) if not overall.empty else float("nan")


def _agg(df: pd.DataFrame, col: str) -> float:
    """Aggregate a metric column across sequences for the summary row.

    Count-based metrics (e.g. ``num_switches``) are summed over the per-sequence
    rows. Every other (ratio/derived) metric uses the pooled, dataset-level
    ``OVERALL`` value, since a plain mean of per-sequence ratios is not the
    correct aggregate for metrics like ``hota`` and ``idf1``. When no pooled row
    is present (older callers), it falls back to the per-sequence mean.
    """
    per_seq = df.loc[df["sequence"] != OVERALL_LABEL, col]
    if col in _SUM_METRICS:
        return per_seq.sum()
    overall = _overall_value(df, col)
    return overall if pd.notna(overall) else per_seq.mean()


def _round_or_none(val: float) -> float | None:
    """Round *val* to two decimal places, or return ``None`` if it is NaN."""
    return round(float(val), 2) if pd.notna(val) else None


def _iter_pf_mn_col(pred_fields: list, metric_names: list, metric_cols: dict) -> chain:
    """Yield ``(pred_field, metric_name, column)`` triples in display order."""
    return chain.from_iterable(
        product([pf], [mn], metric_cols[mn])
        for pf in pred_fields
        for mn in metric_names
    )


def _iter_mn_col(metric_names: list, metric_cols: dict) -> chain:
    """Yield ``(metric_name, column)`` pairs in display order."""
    return chain.from_iterable(product([mn], metric_cols[mn]) for mn in metric_names)


def _summary_values(
    pred_fields: list,
    metric_names: list,
    metric_cols: dict,
    dfs: dict,
) -> dict[tuple[str, str, str], float | None]:
    """Compute pooled summary values once for table and chart consumers."""
    return {
        (pf, mn, col): _round_or_none(_agg(dfs[pf][mn], col))
        for pf, mn, col in _iter_pf_mn_col(pred_fields, metric_names, metric_cols)
    }


def _sortable_headers(
    pf_mn_col: list,
    mn_col: list,
    *,
    show_diff: bool,
    pf_idx: dict,
    n_regular_cols: int,
) -> str:
    """Build sortable column header cells."""
    headers = "".join(
        f'<th class="sortable" data-label="{_h(col)}" data-pf="{pf_idx[pf]}"'
        f' onclick="sortTable({i + 1})" title="Sort by {_h(col)}">{_h(col)}</th>'
        for i, (pf, _mn, col) in enumerate(pf_mn_col)
    )
    if show_diff:
        headers += "".join(
            f'<th class="sortable" data-label="Δ {_h(col)}"'
            f' onclick="sortTable({n_regular_cols + i + 1})"'
            f' title="Sort by Δ {_h(col)}">Δ {_h(col)}</th>'
            for i, (_mn, col) in enumerate(mn_col)
        )
    return headers


def _sequence_rows(
    sequences: list,
    pf_mn_col: list,
    mn_col: list,
    indexed_dfs: dict,
    *,
    show_diff: bool,
    pf_idx: dict,
) -> str:
    """Build per-sequence table body rows."""
    return "".join(
        "<tr><td>"
        + _h(seq)
        + "</td>"
        + "".join(
            f'<td style="text-align:right;" data-pf="{pf_idx[pf]}">'
            f"{_fmt_cell(_cell_value(indexed_dfs[pf][mn], col, seq))}</td>"
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


def _summary_cells(
    pf_mn_col: list,
    mn_col: list,
    summary: dict[tuple[str, str, str], float | None],
    *,
    show_diff: bool,
    pf_idx: dict,
) -> str:
    """Build summary-row metric cells."""
    cells = "".join(
        f'<td style="text-align:right;" data-pf="{pf_idx[pf]}">'
        f"{_fmt_cell(val) if (val := summary[pf, mn, col]) is not None else ''}</td>"
        for pf, mn, col in pf_mn_col
    )
    if show_diff:
        cells += "".join(
            f'<td style="text-align:right;" data-diff-mean'
            f' data-mn="{_h(mn)}" data-col="{_h(col)}"></td>'
            for mn, col in mn_col
        )
    return cells


def _pred_field_headers(
    pred_fields: list, n_cols_per_model: int, pf_idx: dict, *, show_diff: bool
) -> str:
    """Build top-row model name headers."""
    headers = "".join(
        f'<th colspan="{n_cols_per_model}" data-pf="{pf_idx[pf]}"'
        f' style="border-left:2px solid #e94560;">{_h(pf)}</th>'
        for pf in pred_fields
    )
    if show_diff:
        headers += (
            f'<th colspan="{n_cols_per_model}"'
            f' style="border-left:2px solid #e94560;">'
            f'<span id="diff-label">Δ</span></th>'
        )
    return headers


def _metric_name_headers(
    pred_fields: list,
    metric_names: list,
    metric_cols: dict,
    pf_idx: dict,
    *,
    show_diff: bool,
) -> str:
    """Build second-row metric group headers."""
    headers = "".join(
        f'<th colspan="{len(metric_cols[mn])}" data-pf="{pf_idx[pf]}"'
        f' style="border-left:2px solid #0f3460;">'
        f"{_h(DISPLAY_NAMES.get(mn, mn))}</th>"
        for pf in pred_fields
        for mn in metric_names
    )
    if show_diff:
        headers += "".join(
            f'<th colspan="{len(metric_cols[mn])}"'
            f' style="border-left:2px solid #0f3460;">'
            f"{_h(DISPLAY_NAMES.get(mn, mn))}</th>"
            for mn in metric_names
        )
    return headers


def _index_dfs_by_sequence(dfs: dict) -> dict:
    """Return *dfs* with each DataFrame indexed by ``sequence`` once."""
    return {
        pf: {mn: df.set_index("sequence") for mn, df in pf_dfs.items()}
        for pf, pf_dfs in dfs.items()
    }


def _cell_value(indexed: pd.DataFrame, col: str, seq: str) -> float:
    """Look up a single metric value from a sequence-indexed DataFrame."""
    return indexed.reindex([seq]).iloc[0][col]


def _fmt_cell(val: float) -> str:
    """Format a metric value for display in a table cell."""
    return "" if pd.isna(val) else f"{val:.2f}"


def _build_diff_controls(pred_fields: list, show_diff: bool) -> str:
    """Build the diff model-selector HTML, or return empty string."""
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


def _table_layout(pred_fields: list, metric_names: list, metric_cols: dict) -> dict:
    """Pre-compute table iteration order and diff-column flags."""
    return {
        "show_diff": len(pred_fields) >= 2,  # noqa: PLR2004
        "pf_idx": {pf: i for i, pf in enumerate(pred_fields)},
        "n_cols_per_model": sum(len(metric_cols[mn]) for mn in metric_names),
        "pf_mn_col": list(_iter_pf_mn_col(pred_fields, metric_names, metric_cols)),
        "mn_col": list(_iter_mn_col(metric_names, metric_cols)),
    }


def _assemble_html_table(
    sequences: list,
    indexed_dfs: dict,
    summary: dict[tuple[str, str, str], float | None],
    layout: dict,
    *,
    pred_fields: list,
    metric_names: list,
    metric_cols: dict,
) -> str:
    """Assemble the full ``<table>`` HTML string from pre-computed layout values."""
    pf_mn_col = layout["pf_mn_col"]
    mn_col = layout["mn_col"]
    pf_idx = layout["pf_idx"]
    show_diff = layout["show_diff"]
    body_rows = _sequence_rows(
        sequences, pf_mn_col, mn_col, indexed_dfs, show_diff=show_diff, pf_idx=pf_idx
    )
    summary_row = _summary_cells(
        pf_mn_col, mn_col, summary, show_diff=show_diff, pf_idx=pf_idx
    )
    return (
        _build_diff_controls(pred_fields, show_diff)
        + f"""<table id="seq-table">
    <thead>
      <tr><th rowspan="3">Sequence</th>{
            _pred_field_headers(
                pred_fields, layout["n_cols_per_model"], pf_idx, show_diff=show_diff
            )
        }</tr>
      <tr>{
            _metric_name_headers(
                pred_fields, metric_names, metric_cols, pf_idx, show_diff=show_diff
            )
        }</tr>
      <tr>{
            _sortable_headers(
                pf_mn_col,
                mn_col,
                show_diff=show_diff,
                pf_idx=pf_idx,
                n_regular_cols=len(pf_mn_col),
            )
        }</tr>
    </thead>
    <tbody>
      {body_rows}
      <tr id="mean-row" style="font-weight:bold;border-top:2px solid #e94560;">
        <td>OVERALL / SUM</td>{summary_row}
      </tr>
    </tbody>
  </table>"""
    )


def _build_chart_section(
    pred_fields: list,
    metric_names: list,
    metric_cols: dict,
    summary: dict[tuple[str, str, str], float | None],
    color_map: dict,
) -> tuple[dict, str, str, str]:
    """Build the chart-data dict, model checkboxes, tab buttons, and chart grids."""
    chart_data = {
        mn: {
            col: {pf: summary[pf, mn, col] for pf in pred_fields}
            for col in metric_cols[mn]
        }
        for mn in metric_names
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
        f"{_h(DISPLAY_NAMES.get(mn, mn))}</button>"
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


def _overall_data_from_summary(
    summary: dict[tuple[str, str, str], float | None],
    pred_fields: list,
    metric_names: list,
    metric_cols: dict,
) -> dict:
    """Re-nest flat summary values for client-side diff computation."""
    return {
        pf: {
            mn: {col: summary[pf, mn, col] for col in metric_cols[mn]}
            for mn in metric_names
        }
        for pf in pred_fields
    }


def _build_table_html(
    pred_fields: list,
    metric_names: list,
    metric_cols: dict,
    sequences: list,
    dfs: dict,
    *,
    summary: dict[tuple[str, str, str], float | None],
    layout: dict,
) -> tuple[dict, str]:
    """Build the per-sequence table data dict and HTML table string."""
    indexed_dfs = _index_dfs_by_sequence(dfs)
    table_data = {
        pf: {
            mn: {
                col: {
                    seq: _round_or_none(_cell_value(indexed_dfs[pf][mn], col, seq))
                    for seq in sequences
                }
                for col in metric_cols[mn]
            }
            for mn in metric_names
        }
        for pf in pred_fields
    }
    table_html = _assemble_html_table(
        sequences,
        indexed_dfs,
        summary,
        layout,
        pred_fields=pred_fields,
        metric_names=metric_names,
        metric_cols=metric_cols,
    )
    return table_data, table_html


def _comparison_sections(dfs: dict) -> dict:
    """Compute every HTML fragment and JSON payload for the comparison report."""
    pred_fields = list(dfs.keys())
    metric_names = list(next(iter(dfs.values())).keys())
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
        pf: MODEL_COLORS[i % len(MODEL_COLORS)] for i, pf in enumerate(pred_fields)
    }
    summary = _summary_values(pred_fields, metric_names, metric_cols, dfs)
    layout = _table_layout(pred_fields, metric_names, metric_cols)
    chart_data, checkboxes_html, tab_buttons, chart_grids = _build_chart_section(
        pred_fields, metric_names, metric_cols, summary, color_map
    )
    table_data, table_html = _build_table_html(
        pred_fields,
        metric_names,
        metric_cols,
        sequences,
        dfs,
        summary=summary,
        layout=layout,
    )
    return {
        "pred_fields": pred_fields,
        "metric_names": metric_names,
        "sequences": sequences,
        "metric_cols": metric_cols,
        "color_map": color_map,
        "overall_data": _overall_data_from_summary(
            summary, pred_fields, metric_names, metric_cols
        ),
        "chart_data": chart_data,
        "checkboxes_html": checkboxes_html,
        "tab_buttons": tab_buttons,
        "chart_grids": chart_grids,
        "table_data": table_data,
        "table_html": table_html,
    }


def build_comparison_html(dfs: dict) -> str:
    """Build an interactive HTML report with bar charts and a comparison table.

    Args:
        dfs: Nested dict
            ``{pred_field: {"TrackingMetrics": df, "HOTAMetrics": df}}``.
            Each DataFrame has a ``sequence`` column plus numeric metric columns.

    Raises:
        ValueError: If *dfs* is empty.
    """
    if not dfs:
        raise ValueError("dfs must contain at least one element")

    parts = _comparison_sections(dfs)
    template = (pathlib.Path(__file__).parent / "comparison_report.html").read_text(
        encoding="utf-8"
    )

    return (
        template.replace("__SUM_METRICS__", _json_for_html_script(sorted(_SUM_METRICS)))
        .replace("__CHART_DATA__", _json_for_html_script(parts["chart_data"]))
        .replace("__OVERALL_DATA__", _json_for_html_script(parts["overall_data"]))
        .replace("__TABLE_DATA__", _json_for_html_script(parts["table_data"]))
        .replace("__METRIC_COLS__", _json_for_html_script(parts["metric_cols"]))
        .replace("__SEQUENCES__", _json_for_html_script(parts["sequences"]))
        .replace("__PRED_FIELDS__", _json_for_html_script(parts["pred_fields"]))
        .replace("__METRIC_NAMES__", _json_for_html_script(parts["metric_names"]))
        .replace("__COLOR_MAP__", _json_for_html_script(parts["color_map"]))
        .replace("__CHECKBOXES_HTML__", parts["checkboxes_html"])
        .replace("__TAB_BUTTONS__", parts["tab_buttons"])
        .replace("__CHART_GRIDS__", parts["chart_grids"])
        .replace("__TABLE_HTML__", parts["table_html"])
    )
