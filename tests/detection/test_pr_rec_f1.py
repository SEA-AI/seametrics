import numpy as np
from seametrics.detection import PrecisionRecallF1Support

def test_empty_input_with_default_args():
    metric = PrecisionRecallF1Support()
    preds = [
        dict(
            boxes=np.array([]),
            scores=np.array([]),
            labels=np.array([]),
        )
    ]
    target = [
        dict(
            boxes=np.array([]),
            labels=np.array([]),
        )
    ]
    metric.update(preds, target)
    results = metric.compute()
    for key, val in results['metrics'].items():
        assert key == 'all'
        assert val['iouThr'] == '0.50'
        assert val['maxDets'] == 100
        assert val['tp'] == -1
        assert val['fp'] == -1
        assert val['fn'] == -1
        assert val['duplicates'] == -1
        assert val['precision'] == -1
        assert val['recall'] == -1
        assert val['f1'] == -1
        assert val['support'] == 0
        assert val['fpi'] == 0
        assert val['nImgs'] == 1
        
