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

<details>
<summary>Tracking Metrics</summary>

## Tracking Metrics

`TrackingMetrics` wraps [motmetrics](https://github.com/cheind/py-motmetrics) to compute standard MOT scores (MOTA, MOTP, IDF1, …). `HOTAMetrics` implements [HOTA](https://link.springer.com/article/10.1007/s11263-020-01375-2) (Higher Order Tracking Accuracy), which jointly evaluates detection and association quality.

Both classes share the same interface and can be evaluated together in a single dataset pass using `compute_all_metrics_by_sequence`.

```python
import fiftyone as fo
from seametrics.tracking import TrackingMetrics, HOTAMetrics
from seametrics.tracking.utils import compute_all_metrics_by_sequence, results_to_df

dataset = fo.load_dataset("my_dataset")
view = dataset.load_saved_view("my_view")

instances, excluded = compute_all_metrics_by_sequence(
    view=view,
    gt_field="ground_truth_det_fused_id",
    pred_fields=["model_a", "model_b"],
    metrics=[
        (TrackingMetrics, {"max_iou": 0.5}),
        (HOTAMetrics, {}),
    ],
)
```

Returns ``(instances, excluded_sequences)`` where *instances* is a nested dict
``{pred_field: {metric_class_name: metric_instance}}`` and *excluded_sequences*
is the union of hard failures across all models (use it to align comparison
tables). Convert any entry to a per-sequence DataFrame with ``results_to_df``,
passing the same filtered sequence list to every model:

```python
valid = [
    s
    for s in instances["model_a"]["TrackingMetrics"].accumulators
    if s not in excluded
]
mot_df  = results_to_df(instances["model_a"]["TrackingMetrics"], sequence_list=valid)
hota_df = results_to_df(instances["model_a"]["HOTAMetrics"], sequence_list=valid)
```

`TrackingMetrics` DataFrame columns: `sequence`, `num_frames`, `num_unique_objects`, `mota`, `motp`, `idf1`, `idp`, `idr`, `mostly_tracked`, `partially_tracked`, `mostly_lost`, `num_switches`, `num_false_positives`, `num_misses`, `num_fragmentations`, `precision`, `recall`.

`HOTAMetrics` DataFrame columns: `sequence`, `hota`, `deta`, `assa`, `loca`, `num_unique_objects`. Scores are expressed as percentages (0–100).

`num_unique_objects` is included in both DataFrames so you can compute a track-count-weighted global score:

```python
weighted_hota = (
    (hota_df["hota"] * hota_df["num_unique_objects"]).sum()
    / hota_df["num_unique_objects"].sum()
)
```

Hard failures (missing keyframes, missing track IDs, unexpected errors) are logged
rather than raised, and are accessible via ``metric_instance.failed_sequences``.
Empty GT or empty predictions on keyframes are valid evaluation cases (metrics
may be ``NaN``/``-inf`` per motmetrics rules) and are not logged as failures.
Use ``excluded_sequences`` to drop any sequence that failed for any model from
cross-model comparison tables.

</details>
