"""MOT tracking metrics (MOTA/MOTP/IDF1/…) backed by motmetrics."""

import motmetrics as mm
import numpy as np


class TrackingMetrics:
    """MOT metrics wrapper around ``motmetrics`` with per-sequence accumulators."""

    def __init__(self, **kwargs: object) -> None:
        """Initialise accumulators and defaults; extra kwargs become attributes."""
        self.accumulators = {}
        self.max_iou = 0.5
        self.metrics = [
            "num_frames",
            "mota",
            "motp",
            "idf1",
            "idp",
            "idr",
            "mostly_tracked",
            "partially_tracked",
            "mostly_lost",
            "num_switches",
            "num_false_positives",
            "num_misses",
            "num_fragmentations",
            "precision",
            "recall",
            "num_unique_objects",
        ]
        self.failed_sequences = {}
        for key, value in kwargs.items():
            setattr(self, key, value)

    def update(self, gt: np.ndarray, pred: np.ndarray, sequence_name: str) -> None:
        """Build a MOTAccumulator for *sequence_name* from (gt, pred) arrays."""
        num_frames = max(gt[:, 0].max(), pred[:, 0].max()) + 1
        acc = mm.MOTAccumulator(auto_id=True)

        for i in range(1, int(num_frames)):
            gt_dets = gt[gt[:, 0] == i, 1:6]
            pred_dets = pred[pred[:, 0] == i, 1:6]

            dist_matrix = mm.distances.iou_matrix(
                gt_dets[:, 1:], pred_dets[:, 1:], max_iou=self.max_iou
            )
            acc.update(
                gt_dets[:, 0].astype("int").tolist(),
                pred_dets[:, 0].astype("int").tolist(),
                dist_matrix,
            )

        self.accumulators[sequence_name] = acc

    def compute(self, sequence: "str | list | None" = None) -> dict:
        """Compute MOT metrics.

        Parameters
        ----------
        sequence:
            - ``None``: pool all stored sequences and generate an ``OVERALL`` row
              (events are pooled before metrics are computed — MOT standard).
            - ``str``: compute for that single sequence.
            - ``list``/``tuple`` of names: pool exactly that subset and generate an
              ``OVERALL`` row. Pooling is identical to ``None`` over that subset, so
              the per-sequence rows are unchanged.
        """
        mh = mm.metrics.create()
        if sequence is None or isinstance(sequence, (list, tuple)):
            names = (
                list(self.accumulators.keys()) if sequence is None else list(sequence)
            )
            duplicates = sorted({n for n in names if names.count(n) > 1})
            if duplicates:
                raise ValueError(f"Duplicate sequence: {duplicates}")
            unknown = [n for n in names if n not in self.accumulators]
            if unknown:
                raise ValueError(f"Unknown sequence: {unknown}")
            summary = mh.compute_many(
                [self.accumulators[n] for n in names],
                metrics=self.metrics,
                names=names,
                generate_overall=True,
            )
        else:
            if sequence not in self.accumulators:
                raise ValueError(f"Unknown sequence: {sequence}")
            summary = mh.compute(self.accumulators[sequence], metrics=self.metrics)

        return summary.to_dict()

    def log_failed_sequence(
        self, sequence_name: str, gt: list, pred: list, exc: "Exception | None" = None
    ) -> None:
        """Record why *sequence_name* could not be evaluated."""
        if len(gt) == 0 and len(pred) == 0:
            reason = "No ground truth and no predictions"
        elif len(gt) == 0:
            reason = "No ground truth"
        elif len(pred) == 0:
            reason = "No predictions"
        elif exc is not None:
            reason = f"{type(exc).__name__}: {exc}"
        else:
            reason = "Missing IDs from GT or Pred"
        self.failed_sequences[sequence_name] = reason

    @staticmethod
    def metrics_help() -> None:
        """Print the markdown table of metrics supported by motmetrics."""
        print(mm.metrics.create().list_metrics_markdown())
