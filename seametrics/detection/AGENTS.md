# AGENTS.md — `seametrics/detection`

Guidance for producing a per-sequence detection report from a FiftyOne dataset.
Read the repo-root `AGENTS.md` first; this file adds what is specific to detection.

## The recipe

```python
from seametrics.detection.utils import (
    OVERALL_KEY, aggregate_sequence_results,
    payload_to_det_metrics_by_sequence, sequence_results_to_df,
)
from seametrics.payload.processor import PayloadProcessor

payload = PayloadProcessor(
    dataset_name="<dataset>",
    gt_field="<gt field>",
    models=["<model_a>", "<model_b>"],
    slices=["<slice>"],          # e.g. thermal_wide; omit to auto-select by data_type
    sequence_list=batch,         # process in batches, see "Memory" below
    batch_size=8,
).payload

results = payload_to_det_metrics_by_sequence(
    payload,
    model_name="<model_a>",      # one model per call
    keyframes_only=True,         # see "Keyframes"
    iou_thresholds=[1e-9],       # see "IoU threshold"
    include_overall=False,       # pool once at the end instead of per batch
)
```

Accumulate `results` across batches into one dict per model, then:

```python
pooled = dict(per_model_results)
pooled[OVERALL_KEY] = aggregate_sequence_results(pooled)
df = sequence_results_to_df(pooled)     # one row per (sequence, area range)
```

`aggregate_sequence_results` sums the counts and recomputes the ratios once. It is
exact — identical to evaluating every frame in a single metric — because matching
never crosses a frame boundary, so counts are additive.

## Traps

Each of these has produced a wrong report at least once.

### `-1` is "undefined", not a score

`precision`, `recall` and `f1` are `-1` when undefined: no predictions, no ground
truth, or `precision + recall == 0`. Averaging or differencing these columns
silently manufactures nonsense — subtracting `-1` from `0.55` yields a "regression"
of `1.55`, outside the metric's range.

Filter per metric before comparing. The sets differ: F1 is undefined whenever
precision or recall is, **plus** when both are zero, so a sequence can have a
usable precision and no usable F1. Counts (`tp`, `fp`, `fn`, `duplicates`, `fpi`,
`support`, `nImgs`) are never `-1`.

### Pooled is micro, not macro

`OVERALL` sums counts and computes ratios once, so it is support-weighted. It is
**not** the mean of the per-sequence values, and a long sequence outweighs a short
one. A model can lead on pooled recall while trailing on the majority of sequences.
Report both if the audience will read one as the other.

### Boxes are `xywh`

`frame_dets_to_det_metrics` emits absolute-pixel `xywh` because that is FiftyOne's
layout. `PrecisionRecallF1Support` defaults to `box_format="xyxy"`, which silently
mangles the boxes. `payload_to_det_metrics_by_sequence` locks the format and raises
if you pass `box_format`; if you construct the metric yourself, pass `"xywh"`.

### Area is always derived from the bbox

Any `area` on the annotation or in the target dict is ignored — area-range
bucketing uses the same geometry as the IoU. Area ranges are also independent
pools: "small" + "large" does not sum to "all", because ground truth outside a
range is ignored rather than counted.

## IoU threshold

**Never 0.** The match test is `if iou < threshold: continue`, so a threshold of 0
accepts an IoU of 0 — a detection matches a ground truth it does not touch, and
which one it claims comes down to sort order. Every detection in a frame with any
ground truth becomes a true positive.

The useful floor is roughly `1 / (width * height)`: the smallest IoU two genuinely
overlapping boxes can produce is a one-pixel intersection against a frame-sized
union. At 640x512 that is `3.05e-6`, so `1e-5` would reject real overlaps between
badly mismatched box sizes.

Use `1e-9` unless you specifically want strict localisation. It is far below
anything achievable at any sensible resolution, stays strictly positive, and does
not need revisiting when the sensor changes. For far-range thermal targets a few
pixels across, IoU 0.5 fails on a one-pixel offset and measures localisation
tightness rather than whether the model saw the object.

## Keyframes

`PayloadProcessor` records `Sequence.keyframes = {model: [bool, ...]}` when the
prediction field carries a `keyframe` attribute. With `keyframes_only=True` the
mask drops ground truth and predictions together, matching
`seametrics.tracking.utils.build_detection_inputs`. Frames the model never emitted
on leave the evaluation entirely instead of counting as misses, and `nImgs` becomes
the keyframe count.

Two things to handle:

- **Sequences with no keyframes cannot be evaluated this way** and raise. Skip them
  and say so — on `QA_SENTRY_2026_03_VIDEO_BB` this was 16 of 106 sequences, 15 of
  them missing for both models, which pointed at whole campaigns where the tracker
  stage never ran rather than anything about the models.
- **Masks are per prediction field.** Two models with different keyframes are
  scored over different frame subsets, with different `support` and `nImgs`. Each
  number is internally valid; direct comparison is not strictly like-for-like. Pool
  both models over the *union* of skipped sequences so at least the sequence set
  matches, and state the caveat.

## Memory and query limits

`PayloadProcessor` issues one query per field per batch and halves the batch when
MongoDB rejects the result. Two limits are in play:

- **16 MB BSON per result.** A nested query packs a whole sequence into one
  document; dense sequences exceed the cap. The final fallback re-queries a single
  sequence unwound, streaming frames as separate documents.
- **Memory.** Detections are deserialised as `fo.Detection` objects. 25 sequences
  held 2.6 GB resident. Build a payload for a batch, evaluate it, discard it, and
  keep only the per-sequence result dicts — they are small, and pooling at the end
  is exact.

Deserialisation dominates, not round trips: one field across 106 sequences took
~90s regardless of batching. Expect ~10 minutes for a 106-sequence, 3-field run.

## Reporting

`sequence_results_to_df` takes `{sequence_name: results}` and yields one row per
(sequence, area range), so an `OVERALL` entry becomes an OVERALL row. Keep every
evaluated sequence in the table and mark which ones fed the pooled figure, rather
than dropping them — a sequence one model could measure and the other could not is
a finding, not noise.

Always state coverage next to the headline: how many sequences exist, how many were
pooled, how many were excluded and why.
