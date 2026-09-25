# Event Triad Refinement

Validation-locked method: **original**.
Calibration is evaluated out of fold; its final bias is fitted only after that decision.

| Split / Method | Accuracy | Macro-F1 |
|---|---:|---:|
| validation / calibration_oof | 0.758818 +/- 0.005158 | 0.611744 +/- 0.013912 |
| validation / context | 0.768321 +/- 0.003212 | 0.610769 +/- 0.008462 |
| validation / original | 0.767818 +/- 0.002799 | 0.607320 +/- 0.010212 |
| validation / state | 0.769234 +/- 0.002547 | 0.610515 +/- 0.008990 |

## Registered Comparisons

- calibration_oof_minus_original: F1 +0.004423, CI [0.0004980047155990644, 0.008214971340776736]; Accuracy -0.009000; effective=False.
- context_minus_original: F1 +0.003448, CI [0.0019254768528031078, 0.0052285739273428234]; Accuracy +0.000503; effective=False.
- state_minus_context: F1 -0.000254, CI [-0.001894058555894367, 0.0007888086106856854]; Accuracy +0.000914; effective=False.
- state_minus_original: F1 +0.003195, CI [0.0019144166292707422, 0.004501369101596816]; Accuracy +0.001416; effective=False.

## Bidirectional Errors

Three-seed mean counts on the same 7,296 validation events.

| Method | Duel to Pass | Others to Pass | Pass to Duel | Pass to Others |
|---|---:|---:|---:|---:|
| original | 604.33 | 343.33 | 198.33 | 51.33 |
| calibration_oof | 546.33 | 274.33 | 228.33 | 172.00 |
| context | 582.33 | 326.33 | 216.00 | 63.67 |
| state | 586.33 | 328.33 | 208.67 | 61.00 |

Validation admission failed. Original retained; no test data or predictions read.

Other seven probabilities are preserved, but their argmax classifications may change.
Online timings use two validation batches without a dedicated warmed benchmark; they are descriptive, not a speedup claim.
Anchor-state analyses are descriptive; target subevents are not inputs.
Previously seen test split: confirmation on an existing split, not a pristine holdout.
