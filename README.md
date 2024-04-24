<div align="center">
  <p>
    <a align="center" href="https://sea.ai" target="_blank">
      <img width="100%" src="https://github.com/SEA-AI/seametrics/assets/35779409/e685e826-fff5-4ee2-9764-60ff71e047a2"></a>
  </p>
</div>

# <div align="center">seametrics</div>

Library built by SEA.AI to help measure and improve the performance of AI projects.

## <div align="center">Documentation</div>

<details open>
<summary>Install</summary>

```bash
pip install git+https://github.com/SEA-AI/seametrics
```

If you want to test a specific branch
```bash
pip install git+https://github.com/SEA-AI/seametrics@branch-name
```

If you want to install additional dependencies.
```bash
pip install "seametrics[fiftyone] @ git+https://github.com/SEA-AI/seametrics"
```

> For more information about the optional dependencies have a look at the `[project.optional-dependencies]` section of the `pyproject.toml`.

</details>

<details>
<summary>Hugging Face</summary>

Have a look at our [Hugging Face organisation](https://huggingface.co/SEA-AI) to browse through the available metrics.

</details>

<details>
<summary>PrecisionRecallF1Support</summary>

## PrecisionRecallF1Support

Basically a [modified cocoeval.py](https://github.com/SEA-AI/seametrics/blob/develop/seametrics/detection/cocoeval.py) wrapped inside [torchmetrics' mAP metric](https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html) but with numpy arrays instead of torch tensors.

```python
import numpy as np
from seametrics.detection import PrecisionRecallF1Support

predictions = [
    {
        "boxes": np.array(
            [
                [449.3, 197.75390625, 6.25, 7.03125],
                [334.3, 181.58203125, 11.5625, 6.85546875],
            ]
        ),
        "labels": np.array([0, 0]),
        "scores": np.array([0.153076171875, 0.72314453125]),
    }
]

ground_truth = [
    {
        "boxes": np.array(
            [
                [449.3, 197.75390625, 6.25, 7.03125],
                [334.3, 181.58203125, 11.5625, 6.85546875],
            ]
        ),
        "labels": np.array([0, 0]),
        "area": np.array([132.2, 83.8]),
    }
]

metric = PrecisionRecallF1Support() # default settings
metric.update(preds=predictions, target=ground_truth)
metric.compute()['metrics']
```

Will output:
```python
{'all': {'range': [0, 10000000000.0],
  'iouThr': '0.50',
  'maxDets': 100,
  'tp': 0,
  'fp': 2,
  'fn': 2,
  'duplicates': 0,
  'precision': 0.0,
  'recall': 0.0,
  'f1': 0,
  'support': 2,
  'fpi': 0,
  'nImgs': 1}}
```

Where:
- `all` is the area range label
- `range` is the area range
- `iouThr` is the IoU threshold in string format
- `maxDets` is the maximum number of detections
- `tp`, `fp`, `fn` are the true positives, false positives and false negatives
- `duplicates` is the number of duplicates, a duplicate is a prediction that matches an already matched ground truth.
- `precision`, `recall`, `f1` are ... well, the precision, recall and f1 score
- `support` is the number of ground truth boxes
- `fpi` is the false positive index
- `nImgs` is the number of images

</details>
