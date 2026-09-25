# Experiment Status at Export

This file records the status at the snapshot, not the status of a continuously
updated service. The timestamp and live pipeline status at export are in the manifest.

## Stable Baseline

Partial-L2-F80 + Position Head Refit + TC-SoftPred(lambda=1.5).
The supplied three-seed checkpoints correspond to this validated architecture, not
to the unfinished cross-match history experiment. TC is a posterior transform and
has no separate learned checkpoint.

## Coverage and History Study

All six Fixed/Rotate backbone runs completed 24 epochs and 3,192 optimizer steps.
Rotate is locked using full validation (96,891 transitions), three-seed averages,
with Position Refit and TC applied consistently:

| Metric | Original | Rotate |
|---|---:|---:|
| Event Accuracy | 76.64% | 77.91% |
| Event Macro-F1 | 0.6150 | 0.6360 |
| Time MAE (s) | 1.4922 | 1.4577 |
| Position Mean Distance (m) | 15.658 | 14.963 |
| Team Accuracy | 86.77% | 87.30% |
| Player Top-1 | 42.50% | 45.14% |

All seeds improve relative to Original. However, one Fixed seed has no actor-guarded
checkpoint, so no eligible three-seed Rotate-versus-Fixed comparison is available.
These results support improvement over the old baseline, not a clean attribution of
the entire improvement to target coverage alone.

Stage 2 cache creation initially failed because its parent directory was missing.
That implementation issue and report probability-column compatibility were fixed;
18 regression tests and CUDA smoke passed before restarting Stage 2. The ongoing
history study is not a final result and its test gate remains tied to both locks.

## Result Navigation

Experiment protocols and available reports are retained under their original
`paper_1/Phase_1/version_4/experiments/<experiment>/` paths. A directory's presence
does not imply completion: smoke runs, interrupted studies and negative findings
are retained for transparency. Use selection locks and final reports where present.
