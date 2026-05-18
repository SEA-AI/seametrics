import json
import os

import pandas as pd

_SUM_METRICS = {
    "num_frames",
    "mostly_tracked",
    "partially_tracked",
    "mostly_lost",
    "num_switches",
    "num_false_positives",
    "num_misses",
    "num_fragmentations",
    "num_unique_objects",
}

_DISPLAY_NAME = {"TrackingMetrics": "MOT Metrics", "HOTAMetrics": "HOTA Metrics"}

_MODEL_COLORS = [
    "rgba(233,69,96,0.8)",
    "rgba(52,168,235,0.8)",
    "rgba(52,211,153,0.8)",
    "rgba(251,191,36,0.8)",
    "rgba(167,139,250,0.8)",
    "rgba(251,146,60,0.8)",
]


def build_comparison_html(dfs: dict) -> str:
    """Build an interactive HTML report with bar charts and a comparison table.

    Parameters
    ----------
    dfs:
        Nested dict: {pred_field: {"TrackingMetrics": df, "HOTAMetrics": df}}
        Each df has a 'sequence' column plus numeric metric columns.
    """
    pred_fields = list(dfs.keys())
    metric_names = list(next(iter(dfs.values())).keys())

    sequences = sorted(
        next(iter(dfs.values()))[metric_names[0]]["sequence"].tolist()
    )

    metric_cols = {
        m: [c for c in next(iter(dfs.values()))[m].columns if c != "sequence"]
        for m in metric_names
    }

    def _agg(series: pd.Series, col: str):
        return series.sum() if col in _SUM_METRICS else series.mean()

    chart_data = {}
    for mn in metric_names:
        chart_data[mn] = {}
        for col in metric_cols[mn]:
            chart_data[mn][col] = {
                pf: round(float(v), 2)
                if pd.notna(v := _agg(dfs[pf][mn][col], col)) else None
                for pf in pred_fields
            }

    color_map = {pf: _MODEL_COLORS[i % len(_MODEL_COLORS)] for i, pf in enumerate(pred_fields)}

    checkboxes_html = "".join(
        f'<label style="margin-right:16px;cursor:pointer;color:{color_map[pf]};">'
        f'<input type="checkbox" checked onchange="toggleModel(\'{pf}\')" style="margin-right:4px;">'
        f'{pf}</label>'
        for pf in pred_fields
    )

    tab_buttons = "".join(
        f'<button onclick="showTab(\'{mn}\')" id="tab-{mn}" '
        f'style="margin-right:8px;padding:6px 14px;cursor:pointer;'
        f'background:{"#e94560" if i == 0 else "#1a1a2e"};color:#fff;border:1px solid #e94560;border-radius:4px;">'
        f'{_DISPLAY_NAME.get(mn, mn)}</button>'
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
            f'{canvases}</div>'
        )

    table_data = {
        pf: {
            mn: {
                col: {
                    seq: (round(float(v), 2) if pd.notna(v := dfs[pf][mn].set_index("sequence").reindex([seq]).iloc[0][col]) else None)
                    for seq in sequences
                }
                for col in metric_cols[mn]
            }
            for mn in metric_names
        }
        for pf in pred_fields
    }

    show_diff = len(pred_fields) >= 2
    n_regular_cols = len(pred_fields) * sum(len(metric_cols[mn]) for mn in metric_names)

    pf_idx = {pf: i for i, pf in enumerate(pred_fields)}

    sortable_headers = "".join(
        f'<th class="sortable" data-label="{col}" data-pf="{pf_idx[pf]}" onclick="sortTable({i + 1})" title="Sort by {col}">{col}</th>'
        for i, (pf, _, col) in enumerate(
            (pf, mn, col)
            for pf in pred_fields for mn in metric_names for col in metric_cols[mn]
        )
    )
    if show_diff:
        sortable_headers += "".join(
            f'<th class="sortable" data-label="Δ {col}" onclick="sortTable({n_regular_cols + i + 1})" title="Sort by Δ {col}">Δ {col}</th>'
            for i, (_, col) in enumerate(
                (mn, col) for mn in metric_names for col in metric_cols[mn]
            )
        )

    def _val(pf, mn, col, seq):
        return dfs[pf][mn].set_index("sequence").reindex([seq]).iloc[0][col]

    table_rows = "".join(
        "<tr><td>" + seq + "</td>" +
        "".join(
            f'<td style="text-align:right;" data-pf="{pf_idx[pf]}">{"" if pd.isna(v := _val(pf, mn, col, seq)) else f"{v:.2f}"}</td>'
            for pf in pred_fields for mn in metric_names for col in metric_cols[mn]
        ) +
        ("".join(
            f'<td style="text-align:right;" data-diff data-mn="{mn}" data-col="{col}" data-seq="{seq}"></td>'
            for mn in metric_names for col in metric_cols[mn]
        ) if show_diff else "") +
        "</tr>"
        for seq in sequences
    )

    agg_cells = "".join(
        f'<td style="text-align:right;" data-pf="{pf_idx[pf]}">{_agg(dfs[pf][mn][col], col):.2f}</td>'
        for pf in pred_fields for mn in metric_names for col in metric_cols[mn]
    )
    if show_diff:
        agg_cells += "".join(
            f'<td style="text-align:right;" data-diff-mean data-mn="{mn}" data-col="{col}"></td>'
            for mn in metric_names for col in metric_cols[mn]
        )

    n_cols_per_model = sum(len(metric_cols[mn]) for mn in metric_names)
    pred_field_headers = "".join(
        f'<th colspan="{n_cols_per_model}" data-pf="{pf_idx[pf]}" style="border-left:2px solid #e94560;">{pf}</th>'
        for pf in pred_fields
    )
    if show_diff:
        pred_field_headers += f'<th colspan="{n_cols_per_model}" style="border-left:2px solid #e94560;"><span id="diff-label">Δ</span></th>'

    metric_name_headers = "".join(
        f'<th colspan="{len(metric_cols[mn])}" data-pf="{pf_idx[pf]}" style="border-left:2px solid #0f3460;">{_DISPLAY_NAME.get(mn, mn)}</th>'
        for pf in pred_fields for mn in metric_names
    )
    if show_diff:
        metric_name_headers += "".join(
            f'<th colspan="{len(metric_cols[mn])}" style="border-left:2px solid #0f3460;">{_DISPLAY_NAME.get(mn, mn)}</th>'
            for mn in metric_names
        )

    diff_controls = ""
    if show_diff:
        opts_a = "".join(f'<option value="{pf}">{pf}</option>' for pf in pred_fields)
        opts_b = "".join(
            f'<option value="{pf}" {"selected" if i == 1 else ""}>{pf}</option>'
            for i, pf in enumerate(pred_fields)
        )
        diff_controls = (
            '<div style="margin-bottom:10px;font-size:12px;">'
            f'Compare: <select id="diff-a" class="diff-sel" onchange="updateDiff()">{opts_a}</select>'
            f'&nbsp;vs&nbsp;<select id="diff-b" class="diff-sel" onchange="updateDiff()">{opts_b}</select>'
            '</div>'
        )

    table_html = diff_controls + f"""<table id="seq-table">
    <thead>
      <tr><th rowspan="3">Sequence</th>{pred_field_headers}</tr>
      <tr>{metric_name_headers}</tr>
      <tr>{sortable_headers}</tr>
    </thead>
    <tbody>
      {table_rows}
      <tr id="mean-row" style="font-weight:bold;border-top:2px solid #e94560;">
        <td>MEAN / SUM</td>{agg_cells}
      </tr>
    </tbody>
  </table>"""

    template_path = os.path.join(os.path.dirname(__file__), "comparison_report.html")
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    return (
        template
        .replace("__CHART_DATA__", json.dumps(chart_data))
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
