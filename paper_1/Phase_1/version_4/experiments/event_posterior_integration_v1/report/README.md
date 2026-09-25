# Event Posterior Integration

Primary population: all Raw-10 events, including unknown Player.

| Split | Method | Event Accuracy | Macro-F1 |
|---|---|---:|---:|
| validation | base_post | 0.770011 +/- 0.003655 | 0.612169 +/- 0.005238 |
| validation | null | 0.768777 +/- 0.002751 | 0.609919 +/- 0.006664 |
| validation | original | 0.767818 +/- 0.002799 | 0.607320 +/- 0.010212 |

| Validation comparison | Macro-F1 gain | 95% CI | Effective |
|---|---:|---|---|
| null_minus_original | +0.002598 | [+0.000490, +0.004759] | False |
| base_post_minus_null | +0.002251 | [-0.003675, +0.010774] | False |
| base_post_minus_original | +0.004849 | [-0.002050, +0.014687] | False |

Validation selected: EI-Original.
Test confirmation: Not evaluated (validation gate).

Warmed CPU model-only timing and parameter counts: efficiency.csv. HGT runs once per branch (3 layer calls total).

Position Refit and TC-SoftPred (lambda=1.5) are fixed; Event uses the unadjusted Player posterior.
No new test evaluation is authorized when validation fails. Existing test results were previously visible; this is not a pristine holdout.
