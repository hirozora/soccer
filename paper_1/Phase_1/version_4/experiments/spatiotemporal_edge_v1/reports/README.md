# Continuous Spatiotemporal Edge Experiment

Validation-locked method: **base**.

Selection uses post-Refit Position and TC-SoftPred (lambda=1.5) Player outputs.
Bootstrap resamples matches only, with the same draws across configurations and seeds.
See metrics.csv, summary.csv, per-seed confusion matrices and the validation lock for numerical evidence.

No test-based reselection. Failure is specific to adjacent-event scalar gating, not all spatiotemporal propagation.
Next research stage: cross-match historical coverage and causal data interfaces. No further gate expansion.
