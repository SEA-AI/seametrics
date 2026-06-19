"""Backward-compatible re-export shim.

The canonical implementation has moved to ``seametrics.report``.
Import from there in new code.
"""

from seametrics.report.report import (  # noqa: F401
    _agg,
    _build_diff_controls,
    _cell_value,
    _fmt_cell,
    _h,
    _js,
    _round_or_none,
    build_comparison_html,
)

__all__ = ["build_comparison_html"]
