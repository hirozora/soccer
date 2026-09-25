# Unified LEM Loss Repair

## Scope

Only the Unified LEM baseline was retrained. Existing HGT, Seq2Event, and
NMSTPP checkpoints and predictions were retained and referenced by the repaired
summary.

## Repair

The legacy fine-32 event loss averaged weighted per-sample losses without
dividing by the selected sample weights. On the feasibility training set this
reduced the event component to about `0.012`, compared with approximately
`0.40` for time and `0.95` for position.

The repaired objective:

- divides weighted cross-entropy by the sum of selected sample weights;
- uses inverse-square-root frequency weights;
- caps the largest class-weight ratio at 5;
- normalizes the sample-weighted mean weight to 1;
- leaves every other model's historical objective unchanged.

A smoke batch produced Event/Time/Position losses of
`1.006/1.007/1.000`, confirming comparable initial scales.

## Experiment

- Data and sample plan: unchanged feasibility protocol.
- Window: K=80.
- Learning rates: `3e-5`, `1e-4`, `3e-4`.
- Selected learning rate: `3e-4`.
- Maximum epochs: 12; patience: 3.
- Final seeds: `20260715`, `20260716`, `20260717`.

## Result

Unified LEM changed from:

| Metric | Legacy | Repaired |
| --- | ---: | ---: |
| Raw-10 Accuracy | 0.372 | 0.674 |
| Raw-10 Macro-F1 | 0.106 | 0.344 |
| Raw-10 Weighted-F1 | 0.237 | 0.664 |
| Time MAE | 2.232 s | 1.838 s |
| Position distance MAE | 31.316 m | 30.960 m |

The repaired model predicts 13-14 fine classes and 7-8 folded raw classes per
seed, compared with unstable 3-4 raw classes in the legacy run. Collapse is
substantially reduced, although rare-class coverage remains incomplete.

The combined report under `experiments/feasibility/summary_repaired` references
the old result directory for every unchanged model and the repair directory for
Unified LEM. `provenance.json` records this replacement explicitly.
