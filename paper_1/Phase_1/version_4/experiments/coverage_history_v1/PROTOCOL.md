# Coverage and causal cross-match history v1

## Frozen scientific protocol

Semantic V3-D2, F80, Partial-L2, candidate sets and task normalization remain unchanged.
No existing graph, checkpoint or experiment output is overwritten.
Three seeds: 20260715, 20260716, 20260717.

## Stage 1

Fixed and Rotate each train from scratch for exactly 24 epochs and 3192 optimizer steps.
Each epoch contains 34048 samples (128 temporal strata per training match), batch 256.
AdamW: lr 9e-4, weight decay 1e-4, clip 5. No early stopping.
Loss: (0.2 Event + Time + Position + 0.05 Team + 0.4 Player) / 3.
Rotate begins at the original selected target, cycles a fixed shuffled permutation
within each stratum, and covers all 449025 training targets by epoch 16.
Selectors are shared across model seeds; batch slot order matches Fixed.
All new runs use deterministic Torch algorithms and CUBLAS_WORKSPACE_CONFIG=:4096:8
to make repeated CUDA aggregation and exact checkpoint continuation reproducible.

Only full validation (96891 transitions) at epochs 4/8/12/16/20/24 selects checkpoints.
Original Partial-L2 and Five-F80 references are re-evaluated on that population.
Choose minimum raw core loss among epochs satisfying raw Team/Player guards:
at most 1pp below Five-F80 and 0.5pp below Partial-L2, independently per seed.
No qualifying epoch means ineligible. Never substitute an unguarded checkpoint.
Refit each selected Position head on the original 34048 targets with full validation;
TC-SoftPred remains lambda=1.5. Original reuses its already locked postprocessing.

Rotate minus Fixed tests supervision coverage; Fixed minus Original also includes
retraining and checkpoint selection effects. Eligible upgrades must improve against
Original by Event F1 >= .005, Time MAE <= -.01s or Position <= -.25m, with two seeds
improving and a match-paired 95% CI excluding zero. Guards: Event accuracy -1pp,
Event F1 > -.02, Time < +.05s, Position < +.5m, Team/Player accuracy -0.5pp.
Select lowest deployed core loss; differences below 1e-4 prefer Original, Fixed, Rotate.

## Stage 2

Freeze the selected backbone and postprocessing. Train only Event residual heads.
History uses completed England matches with kickoff <= current kickoff minus 24h.
Validation/test history rolls forward, without weight updates. Test graphs remain
unread until BOTH selection locks exist. No current-match final aggregate is used.

Each entity has 13 features: Raw-10 frequencies with 20 pseudo-events from the
eligible-past league distribution, log1p event count, log1p match count, seen.
No-history entities are all zero. Team order is visible anchor actor then opponent.
Player features are weighted by frozen TC posterior, never by actual next player.
Roster mapping uses lineup/bench only. Non-derangeable candidates are zero in BOTH
personal-history versions. Offside counts describe recorded actors, not offside traps.

Null / Team / TeamPlayer / ShuffledPlayer share a zero-final-initialized
103 -> 64 -> GELU -> Dropout(.1) -> 10 residual added to original Event logits.
Full 449025 training and 96891 validation targets, including unknown players.
AdamW lr 3e-4, weight decay 1e-4, batch 1024, clip 5, max 30 epochs, patience 5.
Epoch 0 participates. Require accuracy within 1pp of original, then select by
(-Macro-F1, -Accuracy, CE, epoch). All conditions share initialization and RNG streams.

Team must beat Null and Original by .005 F1 with >=2 improving seeds and positive
CI, without >1pp accuracy loss. TeamPlayer must additionally meet that gain over
Team and significantly beat within-team ShuffledPlayer. Null/Shuffled cannot upgrade.
Highest eligible F1 wins; within .001 prefer Team. Otherwise retain original Event.

## Statistics, access and execution

10000 match-cluster paired bootstrap draws, shared across conditions and seeds.
F1 is computed from summed confusion matrices, not average match F1.
Offside AP and recall at FPR <= .001 are descriptive, never selection overrides.
Test confirms locked winners and necessary controls, never chooses methods.
This is confirmation on a previously inspected split, not a pristine holdout.

Entrypoint: scripts/run_coverage_history.py --stage
verify|coverage|history|select|test|report|pipeline.
Workers use --job reference|train|refit|cache|head|online plus --seed and --mode.
Default max 4 GPU workers, one per GPU, 2 DataLoader workers each; max 3 CPU heads.
Smoke and causal audits precede training. Resume restores optimizer, RNG and sampler.
Background unit: football-hgt-v4-coverage-history. Failed dependencies stop progression.
