import motmetrics as mm
import numpy as np


class TrackingMetrics:
    def __init__(self, **kwargs) -> None:
        self.accumulators = {}
        self.max_iou = 0.9999
        self.metrics = ['num_frames', 'mota', 'motp', 'idf1', 'idp', 'idr', 'mostly_tracked',
                        'partially_tracked', 'mostly_lost', 'num_switches', 'num_false_positives',
                        'num_misses', 'num_fragmentations', 'precision', 'recall', 'num_unique_objects']
        self.failed_sequences = {}
        for key, value in kwargs.items():
            setattr(self, key, value)

    def update(self, gt: np.ndarray, pred: np.ndarray, sequence_name: str) -> None:
        num_frames = max(gt[:, 0].max(), pred[:, 0].max()) + 1
        acc = mm.MOTAccumulator(auto_id=True)

        for i in range(1, int(num_frames)):
            gt_dets = gt[gt[:, 0] == i, 1:6]
            pred_dets = pred[pred[:, 0] == i, 1:6]

            C = mm.distances.iou_matrix(gt_dets[:, 1:], pred_dets[:, 1:], max_iou=self.max_iou)
            acc.update(gt_dets[:, 0].astype('int').tolist(), pred_dets[:, 0].astype('int').tolist(), C)

        self.accumulators[sequence_name] = acc

    def compute(self, sequence: str = None) -> dict:
        mh = mm.metrics.create()
        if sequence is None:
            summary = mh.compute_many(
                list(self.accumulators.values()),
                metrics=self.metrics,
                names=list(self.accumulators.keys()),
                generate_overall=True,
            )
        else:
            if sequence not in self.accumulators:
                raise Exception(f'Unknown sequence: {sequence}')
            summary = mh.compute(self.accumulators[sequence], metrics=self.metrics)

        return summary.to_dict()

    def log_failed_sequence(self, sequence_name: str, gt: list, pred: list, exc: Exception = None) -> None:
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

    def metrics_help(self):
        print(mm.metrics.create().list_metrics_markdown())
