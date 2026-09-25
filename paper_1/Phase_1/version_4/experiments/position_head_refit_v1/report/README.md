# Position Head Refit

All position-valid samples; three fixed seeds.

| Split | Method | Mean distance (m) |
|---|---|---:|
| test | PR-Original | 15.6275 +/- 0.5339 |
| test | PR-Refit | 15.3312 +/- 0.3152 |
| validation | PR-Original | 16.1069 +/- 0.4670 |
| validation | PR-Refit | 15.7942 +/- 0.2443 |

Validation selected: PR-Refit.
Validation improvement and CI (m): 0.312729, [0.2876612802760947, 0.3367174944243308].
Test practical confirmation: True.

The original Position Head is Linear(64,2) plus sigmoid. No Player condition or hidden layer is added.
Prior test results were visible before this experiment; this is not a pristine holdout.
