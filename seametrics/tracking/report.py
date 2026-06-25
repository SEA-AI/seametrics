"""HTML comparison report builder for tracking metrics."""

import html
import json
import pathlib
from itertools import chain, product

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


def _default_compare_pair(pred_fields: list) -> tuple[str, str | None]:
    """Return initial A/B selection for the comparison table."""
    pf_a = pred_fields[0]
    pf_b = pred_fields[1] if len(pred_fields) >= 2 else None  # noqa: PLR2004
    return pf_a, pf_b


def _metric_group_header_block(metric_names: list, metric_cols: dict) -> str:
    """Build one row of metric-group ``<th>`` cells (MOT, HOTA, …)."""
    return "".join(
        f'<th colspan="{len(metric_cols[mn])}" style="border-left:2px solid #0f3460;">'
        f"{_h(_DISPLAY_NAME.get(mn, mn))}</th>"
        for mn in metric_names
    )


def _compare_slot_headers(
    n_cols_per_model: int,
    *,
    name_a: str,
    name_b: str,
) -> str:
    """Build top-row A / B / Δ model name headers."""
    return (
        f'<th colspan="{n_cols_per_model}" id="model-header-a"'
        f' style="border-left:2px solid #e94560;">{_h(name_a)}</th>'
        f'<th colspan="{n_cols_per_model}" id="model-header-b"'
        f' style="border-left:2px solid #2a2a4a;">{_h(name_b)}</th>'
        f'<th colspan="{n_cols_per_model}"'
        f' style="border-left:2px solid #e94560;">'
        f'<span id="diff-label">Δ</span></th>'
    )


def _sortable_headers(mn_col: list) -> str:
    """Build sortable column header cells for A, B, and Δ blocks."""
    headers = []
    col_idx = 0
    for _slot in ("a", "b"):
        for _mn, col in mn_col:
            col_idx += 1
            headers.append(
                f'<th class="sortable" data-label="{_h(col)}"'
                f' onclick="sortTable({col_idx})" title="Sort by {_h(col)}">'
                f"{_h(col)}</th>"
            )
    for _mn, col in mn_col:
        col_idx += 1
        headers.append(
            f'<th class="sortable" data-label="Δ {_h(col)}"'
            f' onclick="sortTable({col_idx})"'
            f' title="Sort by Δ {_h(col)}">Δ {_h(col)}</th>'
        )
    return "".join(headers)


def _slot_cell(
    dfs: dict,
    pf: str | None,
    mn: str,
    col: str,
    seq: str,
) -> str:
    """Format one per-sequence metric cell for slot A or B."""
    if pf is None:
        return ""
    return _fmt_cell(_cell_value(dfs, pf, mn, col, seq))


def _sequence_rows(
    sequences: list,
    mn_col: list,
    dfs: dict,
    *,
    pf_a: str,
    pf_b: str | None,
) -> str:
    """Build per-sequence table body rows for the A/B comparison layout."""
    return "".join(
        "<tr><td>"
        + _h(seq)
        + "</td>"
        + "".join(
            f'<td style="text-align:right;" data-slot="a"'
            f' data-mn="{_h(mn)}" data-col="{_h(col)}" data-seq="{_h(seq)}">'
            f"{_slot_cell(dfs, pf_a, mn, col, seq)}</td>"
            for mn, col in mn_col
        )
        + "".join(
            f'<td style="text-align:right;" data-slot="b"'
            f' data-mn="{_h(mn)}" data-col="{_h(col)}" data-seq="{_h(seq)}">'
            f"{_slot_cell(dfs, pf_b, mn, col, seq)}</td>"
            for mn, col in mn_col
        )
        + "".join(
            f'<td style="text-align:right;" data-diff'
            f' data-mn="{_h(mn)}" data-col="{_h(col)}" data-seq="{_h(seq)}"></td>'
            for mn, col in mn_col
        )
        + "</tr>"
        for seq in sequences
    )


def _summary_slot_cell(
    summary: dict[tuple[str, str, str], float | None],
    pf: str | None,
    mn: str,
    col: str,
) -> str:
    """Format one summary-row cell for slot A or B."""
    if pf is None:
        return ""
    val = summary.get((pf, mn, col))
    return _fmt_cell(val) if val is not None else ""


def _summary_cells(
    mn_col: list,
    summary: dict[tuple[str, str, str], float | None],
    *,
    pf_a: str,
    pf_b: str | None,
) -> str:
    """Build summary-row metric cells for the A/B comparison layout."""
    cells = "".join(
        f'<td style="text-align:right;" data-slot="a" data-mn="{_h(mn)}"'
        f' data-col="{_h(col)}">{_summary_slot_cell(summary, pf_a, mn, col)}</td>'
        for mn, col in mn_col
    )
    cells += "".join(
        f'<td style="text-align:right;" data-slot="b" data-mn="{_h(mn)}"'
        f' data-col="{_h(col)}">{_summary_slot_cell(summary, pf_b, mn, col)}</td>'
        for mn, col in mn_col
    )
    cells += "".join(
        f'<td style="text-align:right;" data-diff-mean'
        f' data-mn="{_h(mn)}" data-col="{_h(col)}"></td>'
        for mn, col in mn_col
    )
    return cells


def _cell_value(dfs: dict, pf: str, mn: str, col: str, seq: str) -> float:
    """Look up a single metric value from the nested data dict."""
    return dfs[pf][mn].set_index("sequence").reindex([seq]).iloc[0][col]


def _fmt_cell(val: float) -> str:
    """Format a metric value for display in a table cell."""
    return "" if pd.isna(val) else f"{val:.2f}"


def _build_diff_controls(pred_fields: list) -> str:
    """Build the A vs B field selectors above the comparison table."""
    opts_a = "".join(
        f'<option value="{_h(pf)}"{" selected" if i == 0 else ""}>{_h(pf)}</option>'
        for i, pf in enumerate(pred_fields)
    )
    none_selected = len(pred_fields) < 2  # noqa: PLR2004
    opts_b = f'<option value=""{" selected" if none_selected else ""}>—</option>'
    opts_b += "".join(
        f'<option value="{_h(pf)}"{" selected" if i == 1 else ""}>{_h(pf)}</option>'
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


def _table_layout(metric_names: list, metric_cols: dict) -> dict:
    """Pre-compute table iteration order for the fixed A/B/Δ layout."""
    mn_col = list(_iter_mn_col(metric_names, metric_cols))
    return {
        "mn_col": mn_col,
        "n_cols_per_model": sum(len(metric_cols[mn]) for mn in metric_names),
    }


def _assemble_html_table(
    sequences: list,
    dfs: dict,
    summary: dict[tuple[str, str, str], float | None],
    layout: dict,
    *,
    pred_fields: list,
    metric_names: list,
    metric_cols: dict,
) -> str:
    """Assemble the full ``<table>`` HTML string from pre-computed layout values."""
    mn_col = layout["mn_col"]
    n_cols = layout["n_cols_per_model"]
    pf_a, pf_b = _default_compare_pair(pred_fields)
    group_hdr = _metric_group_header_block(metric_names, metric_cols)
    body_rows = _sequence_rows(sequences, mn_col, dfs, pf_a=pf_a, pf_b=pf_b)
    summary_row = _summary_cells(mn_col, summary, pf_a=pf_a, pf_b=pf_b)
    return (
        _build_diff_controls(pred_fields)
        + f"""<table id="seq-table">
    <thead>
      <tr><th rowspan="3">Sequence</th>{
            _compare_slot_headers(
                n_cols,
                name_a=pf_a,
                name_b=pf_b or "—",
            )
        }</tr>
      <tr>{group_hdr}{group_hdr}{group_hdr}</tr>
      <tr>{_sortable_headers(mn_col)}</tr>
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
    table_html = _assemble_html_table(
        sequences,
        dfs,
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
        pf: _MODEL_COLORS[i % len(_MODEL_COLORS)] for i, pf in enumerate(pred_fields)
    }
    summary = _summary_values(pred_fields, metric_names, metric_cols, dfs)
    layout = _table_layout(metric_names, metric_cols)
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
