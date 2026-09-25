# Controlled Feasibility Protocol

## Scope

This profile is a lower-cost test of whether HGT is competitive with each
baseline under aligned data and targets. It is not the final evidence for a
superiority claim.

## Fixed data

- Competition and season: England 2017/18.
- Match split: chronological `266/57/57`.
- Window: K=80 for every model.
- Target: the immediate next raw Wyscout event.
- Input stream: unchanged; unsupported targets are masked, not removed.
- Train targets: 128 temporally stratified transitions per match (`34,048`).
- Validation targets: 128 temporally stratified transitions per match (`7,296`).
- Test targets: every transition (`96,854`).
- Sampling seed: `20260701`, fixed independently of model seeds.

For a match with `N` transitions, the sampler divides `[0,N)` into 128
contiguous integer strata and draws one transition from each stratum. The
sampled transition changes only which target contributes to optimization; its
causal K=80 history remains intact.

## Training and selection

- Optimizer: AdamW, weight decay `1e-4`, gradient clip `5.0`.
- Effective batch size: 256 for every model family.
- HGT micro-batch: 256, so no accumulation is needed in the throughput-tuned
  implementation; the optimizer-step batch size remains 256.
- Execution: two loader workers per job and up to three independent jobs per
  GPU. This changes scheduling only, not any statistical or model setting.
- Maximum epochs: 8; early-stopping patience: 2.
- LR search: the same three family-specific candidates as the full protocol.
- LR tuning seed: `20260715`, validation only.
- Final seeds: `20260715`, `20260716`, `20260717`.
- The selected tuning checkpoint supplies seed `20260715`; it is evaluated on
  test without retraining.
- The full test split is read only after LR and checkpoint selection.

There are 18 LR-tuning runs, 12 additional final training runs, and 6 test-only
evaluations of reused checkpoints: 30 unique training runs in total.

## Reporting

Report three-seed mean and standard deviation, per-class event metrics, and the
same 10,000-replicate paired hierarchical bootstrap used by the full protocol.
The bootstrap pairs predictions by seed, match, sample ID, and target mask.

Conclusions must be phrased as feasibility evidence. A final paper comparison
still requires the full-sample, five-seed protocol or an explicitly justified
follow-up power analysis.
