# Rotate Next-Subevent Auxiliary Supervision

## Scope

SA-Base reuses Rotate. SA-Coarse and SA-Fine are trained from scratch with
identical Partial-L2-F80 public initialization and an identical Linear(64,14)
auxiliary head (910 parameters). Neither ECA nor inference-time probability
aggregation is used. The auxiliary head is removed for deployment.

Raw successor (eventId, subEventId) labels, without tag overrides:

- Duel: (1,10), (1,11), (1,12), (1,13).
- Others: (7,70), (7,71), (7,72).
- Pass: (8,80) through (8,86).

Other coarse classes are masked from auxiliary supervision only. Unknown or
inconsistent subtypes in these three classes fail validation. Unknown-player
samples remain eligible. Labels are targets, never context features.

## Objective

All 14 logits participate in both objectives. Coarse uses CE of the three
logsumexp child groups. Fine uses ordinary 14-way CE. Both are divided by
log(14) and averaged over eligible samples. Empty batches give differentiable
zero. Fine equals Coarse plus within-parent CE / log(14).

Total objective = unchanged Rotate five-task objective + 0.02 * auxiliary loss.
The auxiliary coefficient is outside the original /3. No class weighting or
coefficient search is allowed. Original five-task outputs, not total training
loss, must equal the baseline at initialization.

## Training And Selection

Seeds 20260715-20260717; 24 epochs without early stopping; batch 256;
AdamW LR 9e-4, weight decay 1e-4, gradient clipping 5. Rotate's 34,048 targets
per epoch, exact target plan and match/slot ordering are reused: 3,192 steps
and all 449,025 training targets covered by epoch 16.

Only epochs 4/8/12/16/20/24 use the full 96,891 validation transitions for
checkpoint selection. Raw Team Accuracy and Player Top-1 may fall at most
1pp against Five-F80 and 0.5pp against Partial-L2-F80, per seed. The guarded
checkpoint minimizes original core E/T/P loss. Original joint loss determines
best_joint. Auxiliary metrics never select checkpoints. No guarded epoch means
ineligible, with no fallback. Save last, optimizer, RNG, sampler and history.

Each selected checkpoint supplies its own frozen contexts for the existing
130-parameter Position Refit (fixed 34,048 train, full validation). Player
receives TC-SoftPred with lambda=1.5, without retuning.

## Decisions

Pre-registered comparisons: Coarse-Base, Fine-Coarse, Fine-Base. Event benefit
requires Macro-F1 >= +0.005, at least two improving seeds, and a positive lower
95% confidence bound from 10,000 paired match-cluster bootstrap draws. All
models and seeds share draws; F1 uses pooled confusion counts per seed.

Against Base: Accuracy loss <=1pp, Macro-F1 loss <0.02, Time increase <0.05s,
Position increase <0.50m, Team and TC Player accuracy loss <=0.5pp. Fine also
requires Accuracy loss <=1pp against Coarse. Coarse must effectively improve
Base; Fine must effectively improve both Base and Coarse.

Choose minimum postprocessed validation core loss from eligible candidates,
always including Base. Differences <1e-4 favor Base, then Coarse, then Fine.
Write selection/subevent_auxiliary_lock.json before any test access. Retaining
Base ends the experiment without new test evaluation. Otherwise evaluate the
locked method and required controls on 96,854 test transitions, with no test
reselection. This is confirmation on a previously examined split.

## Reporting And Execution

Report original/postprocessed five-task metrics, seed variation, all Raw-10
per-class scores and confusion, Duel/Others-to-Pass and reverse mistakes,
Pass+CONTROL groups, subtype coverage and auxiliary confusion/accuracy.
Finer supervision helping does not establish it as the unique bottleneck.

Entry: scripts/run_subevent_auxiliary.py --stage
verify|smoke|train|refit|select|test|report|pipeline.
Background: scripts/launch_subevent_auxiliary_background.sh.
At most four GPU workers, one per device, two DataLoader workers per run.
No automatic hierarchy aggregation, ECA rerun or follow-up model search.
