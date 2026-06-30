# Tracking metrics: `ground_truth_det_fused_id` + empty-scene evaluation

## Goal

Evaluate tracker metrics on all meaningful sequences, including confirmed-empty scenes, using `ground_truth_det_fused_id` as GT. Keep fair cross-model comparison (same sequence set per comparison).

## GT field change

- Callers pass `gt_field="ground_truth_det_fused_id"` to `compute_all_metrics_by_sequence`.
- No structural change to `build_detection_inputs` (same `Detections` schema).

## Root cause

`prepare_data_for_det_metrics` returns 1D `np.array([])` when a side has no rows. `TrackingMetrics.update` then raises `IndexError` on `gt[:, 0]` / `pred[:, 0]` → caught and mis-logged as failure via `failed_sequence_reason` (empty GT/pred strings).

## Code fixes (seametrics)

### 1. Empty-array shape (`prepare_data_for_det_metrics`)

```python
# instead of: return np.array(target), np.array(preds)
return (
    np.array(target) if target else np.empty((0, 10)),
    np.array(preds) if preds else np.empty((0, 10)),
)
```

Update `test_prepare_data_none_track_id_skipped` (currently asserts `(0,)`).

### 2. One-sided-empty = compute, not fail

After fix #1, these cases reach motmetrics normally. Do not add them to `failed_sequences`.

| Case | MOT (TrackingMetrics) | HOTA |
|---|---|---|
| No GT, has preds | `num_false_positives=N`, `precision=0`, `recall=NaN`, `mota=-inf` | all scores `0.0` |
| Has GT, no preds | `num_misses=N`, `recall=0`, `precision=NaN`, `mota=0` | all scores `0.0` |
| Both empty | ratio metrics `NaN`, counts `0` | all scores `NaN` |

Reserve `failed_sequences` for hard failures only: missing keyframes, missing track IDs, unexpected exceptions.

Numeric ground truth: `TestTrackingNoGT`, `TestTrackingEmptyPred`, `TestHOTANoGT` / `TestHOTANoPred`. Add explicit both-empty MOT test.

### 3. Fair comparison exclusion (keep policy, fix implementation)

**Keep:** if any model/metric fails on a sequence → exclude from **all** models in comparison.

**Fix:** separate concerns — do not use `log_failed_sequence(seq, [], [], exc=None)` for propagation (overwrites real reasons with `"No ground truth and no predictions"`).

```python
instances, excluded_sequences = compute_all_metrics_by_sequence(...)  # new return
valid = [s for s in sequences if s not in excluded_sequences]

# same valid list for every model — required for fair OVERALL pooling
df_a = results_to_df(results["model_a"]["TrackingMetrics"], sequence_list=valid)
df_b = results_to_df(results["model_b"]["TrackingMetrics"], sequence_list=valid)
```

`excluded_sequences` = union of all `failed_sequences` across every `pred_field × metric_class` instance.

`report.py` sequence intersection is not sufficient alone — OVERALL rows pool only over the `sequence_list` passed to each `results_to_df`.

## Expected outcome on QA_SENTRY failing sequences

| Sequence | fused_id GT (keyframes) | Result |
|---|---|---|
| `...2023_09_29...` | 0 GT, ~108 preds | Compute: FP noise test |
| `...2024_03_BSH...` | 0 GT, 0–2 preds | Compute |
| `...2023_09_28...` | 337 GT, 0 preds | Compute: tracker suppressed all |
| `PROACT_CELADON...` | 128 GT, varies by model | Compute per model; exclude only on genuine per-model failure |

## Tests + docs

- `prepare_data_for_det_metrics`: empty sides return `(0, 10)`.
- Integration: one-sided-empty → metrics, not `failed_sequences`.
- Propagation: union exclusion via `excluded_sequences`; per-instance reasons preserved.
- Update README + AGENTS.md: `failed_sequences` = hard failures only; empty scenes are computed rows with NaN/0/-inf.

## Out of scope

- Per-frame diagnostics: `field=None` vs `detections=[]` in warnings.
- Changing motmetrics / HOTA NaN semantics.
