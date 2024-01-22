import motmetrics as mm
import numpy as np

class TrackingMetrics:
    def __init__(self, **kwargs) -> None:
        self.accumulators = {}  
        self.max_iou = 0.5
        self.metrics = ['num_frames', 'mota', 'motp', 'idf1', 'idp', 'idr', 'mostly_tracked',
                        'partially_tracked', 'mostly_lost', 'num_switches', 'num_false_positives',
                        'num_misses', 'num_fragmentations', 'precision', 'recall', 'num_unique_objects']

        for key, value in kwargs.items():
            setattr(self, key, value)
        
    def update(self, gt: np.ndarray, pred: np.ndarray, sequence_name: str) -> None:
        num_frames = max(gt[:, 0].max(), pred[:, 0].max())+1
        acc = mm.MOTAccumulator(auto_id=True)

        for i in range(1, int(num_frames)):
            gt_dets = gt[gt[:, 0] == i, 1:6]
            pred_dets = pred[pred[:, 0] == i, 1:6]

            C = mm.distances.iou_matrix(gt_dets[:,1:], pred_dets[:,1:], max_iou=self.max_iou)
            acc.update(gt_dets[:,0].astype('int').tolist(), pred_dets[:,0].astype('int').tolist(), C)

        self.accumulators[sequence_name] = acc

    def compute(self, sequence: str = None) -> dict:
        mh = mm.metrics.create()
        if sequence not in self.accumulators:
            raise Exception(f'Unknown sequence: {sequence}')

        if sequence is None:
            summary = mh.compute_many(self.accumulators.values(), metrics=self.metrics, names=self.accumulators.keys(), generate_overall=True)
        else:
            summary = mh.compute(self.accumulators[sequence], metrics=self.metrics)

        return summary.to_dict()
    
    def metrics_help(self): 
        print(mm.metrics.create().list_metrics_markdown())