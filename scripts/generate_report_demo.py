#!/usr/bin/env python3
"""Generate tracking comparison HTML from real metric computation.

Usage:
    python scripts/generate_report_demo.py before  output.html   # develop behaviour
    python scripts/generate_report_demo.py after   output.html   # this branch

Scenario:
  seq_easy — 1 object, 3 frames, tracked perfectly.
  seq_hard — 5 objects × 5 frames; predictions only appear in frame 1 (rest missed).

For model_b the per-sequence HOTA mean (~60%) looks tolerable, but the pooled
dataset score (~38%) reflects that most object-frame associations fail.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from seametrics.tracking.hota import HOTAMetrics
from seametrics.tracking.report import build_comparison_html
from seametrics.tracking.track import TrackingMetrics
from seametrics.tracking.utils import OVERALL_LABEL, results_to_df

if TYPE_CHECKING:
    import pandas as pd

_SEQS = ("seq_easy", "seq_hard")
_EXPECTED_ARGC = 3  # script name + variant + output path


def _det(
    frame: int, obj_id: int, x1: int = 0, y1: int = 0, x2: int = 10, y2: int = 10
) -> list:
    return [frame, obj_id, x1, y1, x2, y2]


def _array(*rows: list) -> np.ndarray:
    return np.array(rows, dtype=float) if rows else np.empty((0, 6))


def _seq_easy_perfect() -> tuple[np.ndarray, np.ndarray]:
    """One object, three frames, flawless tracking."""
    gt = _array(_det(1, 1), _det(2, 1), _det(3, 1))
    return gt, gt.copy()


def _seq_hard_many_poor() -> tuple[np.ndarray, np.ndarray]:
    """Five objects across five frames; only frame-1 detections (rest are misses)."""
    gt = _array(
        *[_det(frame, obj_id) for obj_id in range(1, 6) for frame in range(1, 6)]
    )
    pred = _array(*[_det(1, obj_id) for obj_id in range(1, 6)])
    return gt, pred


def _seq_hard_perfect() -> tuple[np.ndarray, np.ndarray]:
    gt = _array(
        *[_det(frame, obj_id) for obj_id in range(1, 6) for frame in range(1, 6)]
    )
    return gt, gt.copy()


def _model_metrics(
    seq_data: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, pd.DataFrame]:
    hota = HOTAMetrics()
    mot = TrackingMetrics()
    for name, (gt, pred) in seq_data.items():
        hota.update(gt, pred, name)
        mot.update(gt, pred, name)
    return {
        "TrackingMetrics": results_to_df(mot, sequence_list=list(_SEQS)),
        "HOTAMetrics": results_to_df(hota, sequence_list=list(_SEQS)),
    }


def demo_dfs(*, variant: str) -> dict:
    """Nested dfs from real metrics; ``before`` omits the pooled OVERALL row."""
    model_a = _model_metrics(
        {"seq_easy": _seq_easy_perfect(), "seq_hard": _seq_hard_perfect()}
    )
    model_b = _model_metrics(
        {"seq_easy": _seq_easy_perfect(), "seq_hard": _seq_hard_many_poor()}
    )
    dfs = {"model_a": model_a, "model_b": model_b}
    if variant == "before":
        dfs = {
            pf: {
                mn: df.loc[df["sequence"] != OVERALL_LABEL].reset_index(drop=True)
                for mn, df in metrics.items()
            }
            for pf, metrics in dfs.items()
        }
    return dfs


def main() -> None:
    """Parse CLI args and write the demo report to the given path."""
    if len(sys.argv) != _EXPECTED_ARGC or sys.argv[1] not in {"before", "after"}:
        raise SystemExit(f"Usage: {sys.argv[0]} before|after OUTPUT.html")

    variant, out_path = sys.argv[1], Path(sys.argv[2])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        build_comparison_html(demo_dfs(variant=variant)), encoding="utf-8"
    )
    print(f"Wrote {variant} demo -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
