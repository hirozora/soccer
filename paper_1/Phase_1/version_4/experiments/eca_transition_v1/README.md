# Rotate Possession-transition ECA

Independent experiment. No existing graph, checkpoint, or result is overwritten.

## Preregistered Protocol

Base: completed Rotate Partial-L2-F80. Train Constant and Transition for seeds
20260715, 20260716, 20260717. Only Main L2 changes. The 660-parameter controller
is Linear(36,16), GELU, Linear(16,4); its final layer starts at zero. Bias is
added after HGT score scaling and before the unchanged incoming-edge softmax.
Existing same-period next/gap edges share the same pair bias. Other edges have
zero bias. No relation, window, hub, or Player-branch architecture changes.

Feature order: geometry/time (5), team (2), possession identity validity (3),
source/target control one-hots (8), source/target role one-hots (12), switches
(2), source duration/count then target duration/count (4). Statistics use only
selected F80 events up to each endpoint, not anchor summaries. Constant zeros
all 36 inputs including masks. Coordinates use the destination actor's view.

Reuse Rotate's exact target plans, initialization and batch-slot order:
34048 targets/epoch, 24 epochs, 3192 steps, AdamW lr=9e-4, wd=1e-4, batch=256,
clip=5, no early stopping. Existing five-task loss and normalization unchanged.
Full 96891-sample validation only at epochs 4/8/12/16/20/24 selects checkpoints.
Guard raw Team Accuracy and Player Top-1 against same-seed Five-F80 (1pp) and
Partial-L2-F80 (0.5pp), then minimize raw core loss. No guarded checkpoint means
ineligible, never an unguarded fallback. Independently refit each selected
backbone's 130-parameter Position head; apply fixed TC lambda=1.5.

## Validation Selection

Report Constant-Base, Transition-Constant, Transition-Base after identical
postprocessing, plus raw metrics. Event gain requires mean Macro-F1 >=0.005,
two improving seeds and a positive lower 95% CI. Bootstrap: 10000 match-only
draws shared across configurations/seeds; aggregate confusion matrices per
draw, then average seed differences. No event or independent seed resampling.

Base guards: Accuracy drop <=0.01, Macro-F1 drop <0.02, Time increase <0.05s,
Position increase <0.50m, Team/TC-Player drop <=0.005. Transition must beat both
Base and Constant and lose no more than 0.01 Accuracy to Constant. Choose
minimum postprocessed validation core loss among eligible configs plus Base;
ties within 1e-4 prefer Base, Constant, Transition. Constant-only success does
not prove conditional feature value.

No test reads before selection/eca_lock.json. Base retained means no new test
evaluation. Otherwise test only the locked method and necessary controls on
96854 samples, without reselection. This is an existing-split confirmation,
not a pristine blind holdout.

## Execution

Entry: scripts/run_eca_transition.py --stage
verify|smoke|train|refit|select|test|report|pipeline.
Workers accept --mode constant|transition --seed --device --resume.

Launcher: bash scripts/launch_eca_transition_background.sh
User service: football-hgt-v4-eca-transition; requires Linger=yes.
Four GPUs, at most one training worker per GPU, two loader workers per job.
Independent logs and PID/status JSON are in background/. Failed workers retry
once with checkpoint restore; dependent stages stop on persistent failure.

Tests and CUDA smoke precede training. Diagnostics include confusion classes,
Pass+CONTROL versus other anchors, per-head bias and attention mass, and
CPU/H2D/forward/total latency, memory and parameters. Attention diagnostics are
descriptive, not a search criterion. Low-cost claims require <=1.20x forward
and <=1.10x peak memory on the same batch. No automatic hub/ECA expansion.
