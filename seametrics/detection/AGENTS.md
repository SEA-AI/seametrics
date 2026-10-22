# AGENTS.md: `seametrics/detection`

How to produce a per-sequence detection report from a FiftyOne dataset. Read the
repo-root `AGENTS.md` first.

## The recipe

```python
from seametrics.detection.report import EvaluationOptions, evaluate_models

report = evaluate_models(
    dataset_name="<dataset>",
    gt_field="<gt field>",
    models=["<model_a>", "<model_b>"],
    slices=["<slice>"],
    options=EvaluationOptions(keyframes_only=True),
)
df = report.to_df()            # one row per (model, sequence, area range) + OVERALL
report.excluded_sequences      # sequences some model could not evaluate
```

`evaluate_models` fetches a batch of sequences, evaluates every model on it, drops
it, and pools at the end. Pooling covers only the sequences every model evaluated.
`to_df` keeps every evaluated sequence and flags `in_overall`: a sequence one model
could measure and another could not is a finding, not noise. State coverage next
to the headline: sequences in the dataset, pooled, excluded and why.

## Traps

### `-1` is "undefined", not a score

`precision` is `-1` with no predictions, `recall` with no ground truth, and `f1`
when either is `-1` or both are `0`. Subtracting `-1` from `0.55` reports a
"regression" of `1.55`, so filter per metric before comparing. Counts are never
`-1`.

### Pooled is micro, not macro

`OVERALL` sums counts and computes ratios once, so a long sequence outweighs a
short one. A model can lead on pooled recall while trailing on most sequences.
Report both if readers could take one for the other.

### Boxes are `xywh`

`frame_dets_to_det_metrics` emits pixel `xywh`; `PrecisionRecallF1Support`
defaults to `xyxy`. Pass `box_format="xywh"` when you build the metric yourself.

### Area comes from the box

Any `area` input is ignored. Area ranges are separate pools: "small" + "large" is
not "all", because ground truth outside a range is ignored, not counted.

## IoU threshold

Never 0: matching skips only `iou < threshold`, so at 0 a detection matches ground
truth it does not touch. For pixel-aligned boxes the smallest overlap is about
`1 / (width * height)` (`3.05e-6` at 640x512), so `1e-5` already rejects some. The default `1e-9` counts
any overlap at any sensible resolution. IoU 0.5 fails a few-pixel thermal target on
a one-pixel offset, measuring localisation rather than detection.

## Keyframes

With `keyframes_only=True`, frames a model did not flag leave the evaluation for
ground truth and predictions alike, as in `seametrics.tracking`, and `nImgs` counts
keyframes. A sequence with no mask for a model raises; `evaluate_models` skips it
for that model and excludes it from pooling. Masks are per model, so two models are
scored on different frames: each number is valid, the comparison is not
like-for-like. Say so.

## Memory

`fo.Detection` objects are heavy, so keep one batch in memory and only the
per-sequence result dicts across batches.

## Browsing TP / FP / FN in the app

`evaluate_models(..., eval_key="ev")` also runs FiftyOne's `evaluate_detections`
as `ev_<model>` on the same sequences, keyframes, labels and IoU. Every prediction
and ground-truth box is marked `tp`, `fp` or `fn`, so false negatives are
browsable too. FiftyOne's COCO matching agrees box for box with this package's
`"all"` area range; `evaluate_models` warns if its counts ever differ. It has no
area ranges, so it cannot describe the others. Off by default because it writes
to the dataset; a rerun with the same key replaces the evaluation.
