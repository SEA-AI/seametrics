#!/usr/bin/env python3
"""Generate tracking comparison HTML from real metric computation.

Usage:
    python scripts/generate_report_demo.py before  output.html   # develop behaviour
    python scripts/generate_report_demo.py after   output.html   # this branch

Scenario (hand-built MOT arrays, same as test_hota pooling tests):
  - seq_short: 2 frames, 1 object, prediction matches GT on both frames
  - seq_long:  2 frames, 1 object
      model_a: perfect tracking
      model_b: misses the object in frame 2

Per-sequence HOTA for model_b is 100% then 50%. Pooled HOTA is ~79% (not 75% mean).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from seametrics.tracking.hota import HOTAMetrics
from seametrics.tracking.report import build_comparison_html
from seametrics.tracking.track import TrackingMetrics
from seametrics.tracking.utils import OVERALL_LABEL, results_to_df

_SEQS = ("seq_short", "seq_long")


def _det(frame: int, obj_id: int, x1: int, y1: int, x2: int, y2: int) -> list:
    return [frame, obj_id, x1, y1, x2, y2]


def _array(*rows: list) -> np.ndarray:
    return np.array(rows, dtype=float)


def _perfect_track() -> tuple[np.ndarray, np.ndarray]:
    gt = _array(_det(1, 1, 0, 0, 10, 10), _det(2, 1, 0, 0, 10, 10))
    return gt, gt.copy()


def _one_frame_miss() -> tuple[np.ndarray, np.ndarray]:
    gt = _array(_det(1, 1, 0, 0, 10, 10), _det(2, 1, 0, 0, 10, 10))
    pred = _array(_det(1, 1, 0, 0, 10, 10))
    return gt, pred


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
        {"seq_short": _perfect_track(), "seq_long": _perfect_track()}
    )
    model_b = _model_metrics(
        {"seq_short": _perfect_track(), "seq_long": _one_frame_miss()}
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
    if len(sys.argv) != 3 or sys.argv[1] not in {"before", "after"}:
        raise SystemExit(f"Usage: {sys.argv[0]} before|after OUTPUT.html")

    variant, out_path = sys.argv[1], Path(sys.argv[2])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_comparison_html(demo_dfs(variant=variant)), encoding="utf-8")
    print(f"Wrote {variant} demo -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
