# AGENTS.md: `seametrics/detection`

How to read a detection report. The code documents how it is computed.

- **`-1` is "undefined", not a score.** Subtracting `-1` from `0.55` reports a
  "regression" of `1.55`, so filter per metric before comparing.
- **`OVERALL` is micro, not macro.** A long sequence outweighs a short one, so a
  model can lead on pooled recall while trailing on most sequences. Report both
  if readers could take one for the other.
- **Keyframe masks are per model.** With `keyframes_only=True` two models are
  scored on different frames: each number is valid, the comparison is not
  like-for-like. Say so.
- **State coverage next to the headline:** sequences in the dataset, pooled,
  excluded and why (`report.excluded_sequences`). A sequence one model could
  measure and another could not is a finding, not noise.
