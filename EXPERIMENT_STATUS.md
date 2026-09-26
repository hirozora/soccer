# Experiment Status (2026-09-26)

## Current Baseline

Rotate-trained Semantic V3-D2 / Partial-L2-F80, with per-backbone Position Head
Refit and TC-SoftPred (lambda=1.5). Coverage/history and ECA pipelines completed.
The older downloadable archive and its manifest are an immutable prior snapshot;
its original Partial-L2 checkpoints are not the newly selected Rotate checkpoints.

## Coverage and History

Rotate was selected on full validation. Test confirmation (three-seed means):

| Metric | Original | Rotate |
|---|---:|---:|
| Event Accuracy | 76.854% | 78.160% |
| Event Macro-F1 | 0.61545 | 0.63187 |
| Time MAE (s) | 1.49232 | 1.46320 |
| Position Error (m) | 15.331 | 14.683 |
| Team Accuracy | 87.095% | 87.678% |
| TC Player Top-1 | 42.482% | 45.210% |

One Fixed seed failed actor guards, so there is no eligible three-seed
Rotate-versus-Fixed causal comparison. Improvement over Original cannot all be
attributed solely to supervision coverage. Stage 2 history priors did not qualify
for an Event upgrade; the selected Rotate Event head is unchanged.

## Possession-transition ECA

Six runs completed 24 epochs / 3192 steps, with Position Refit and fixed TC.
Full-validation means (96891 samples):

| Metric | Rotate Base | Constant | Transition |
|---|---:|---:|---:|
| Event Accuracy | 77.9109% | 77.9646% | 77.9529% |
| Event Macro-F1 | 0.636026 | 0.635415 | 0.635336 |
| Time MAE (s) | 1.457720 | 1.456253 | 1.453539 |
| Position Error (m) | 14.963284 | 14.950680 | 14.894102 |
| Team Accuracy | 87.3033% | 87.3077% | 87.4729% |
| TC Player Top-1 | 45.1398% | 45.1708% | 45.1857% |

Transition minus Base Event F1: -0.000690, match-cluster paired 95% CI
[-0.002248, 0.000792]. Transition minus Constant: -0.000080. Neither qualifies.
Pass+CONTROL forward Duel/Others-to-Pass errors fall from 6258.7 to 6212.0 per
seed, but reverse errors rise from 818.7 to 872.7; group error rate does not improve.
Base remains locked; no new ECA test evaluation was performed. Failure applies
to this Main-L2 adjacent-event bias design, not to all edge-aware attention.

## Navigation

Completed protocols, selection locks and reports:

- paper_1/Phase_1/version_4/experiments/coverage_history_v1/
- paper_1/Phase_1/version_4/experiments/rotate_event_error_audit_v1/
- paper_1/Phase_1/version_4/experiments/eca_transition_v1/

Validation selects methods; previously examined test splits only confirm them.
Raw data, graph tensors, feature caches, per-sample predictions and large model
checkpoints are excluded from this incremental publication.

## ECA Anchor/Age Audit (Completed)

Frozen-checkpoint analysis covers all 96,891 validation samples for three seeds.
No retraining or test access. Transition suppresses adjacent-event attention
throughout the age range, not preferentially on old pairs. For Pass+CONTROL,
attention mass on the pair ending at the anchor changes from 0.1809 (Base) to
0.1154 (Transition); other anchors change from 0.2580 to 0.2343.
These are descriptive comparisons across separately trained models, not isolated
causal effects or proof of feature redundancy.
See experiments/eca_transition_v1/diagnostics/anchor_age_v1/ under version_4.

## Next-Subevent Auxiliary Supervision (Running)

The independent subevent_auxiliary_v1 pipeline started on 2026-09-26 (JST).
It compares Rotate Base with Coarse and Fine auxiliary supervision using the
same 14-output linear head and fixed auxiliary coefficient 0.02.
Six 24-epoch runs, three seeds; no ECA and no inference-time fine/coarse fusion.

15 unit tests, full train/validation subtype audit, exact Rotate sampling-plan
checks and CUDA smoke passed. Initial five-task outputs and original losses
match the baseline exactly. The auxiliary training objective is intentionally
different. Auxiliary eligible targets: 388,101 train and 83,662 validation.
Four GPU workers run under an independent user systemd service with Linger=yes.

No final auxiliary results or validation decision are available in this snapshot.
Position Refit, validation bootstrap/lock, conditional test and reporting follow
automatically. Model selection is validation-only; retaining Base skips new test
evaluation. Current official baseline remains Rotate + Position Refit + fixed TC.
Raw data, caches, in-flight model checkpoints and large target-plan records are
not included in this publication.
