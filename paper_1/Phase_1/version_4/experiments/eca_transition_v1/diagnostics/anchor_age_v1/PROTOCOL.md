# Read-only Anchor/Age Attention Audit

Existing guarded-core checkpoints only: Rotate Base, ECA Constant, ECA Transition;
seeds 20260715-20260717. Full validation, no training and no test reads. This
post-hoc diagnostic never changes model selection or the official baseline.

Pair age = anchor local Event rank minus pair destination local rank. Buckets:
0 (ends at anchor), 1-5, 6-20, greater than 20. Cross with anchor Pass+CONTROL
versus other anchors. Same-period controlled next/gap edges only; record the
number of available pairs and samples for each bucket, excluding period breaks.

Adjacent attention mass sums the existing next and corresponding gap edge
weights for each unique pair, after HGT normalization over ALL incoming edges.
It is not renormalized over adjacent edges. Report each of four heads and the
head mean. Primary: mean per-pair mass within each sample/bucket, then equal
sample weighting. Secondary: pair-weighted mean. Repeated overlapping windows
are not treated as independent observations for inference.

Report three-seed means, seed standard deviations, learned bias means and
10000 match-cluster paired bootstrap CIs with identical match draws across
models, age groups and seeds. Descriptive contrasts: the adjustment to focal
anchor-ending versus focal old pairs; focal versus other anchor-ending pairs;
their age-by-anchor interaction. These do not add an upgrade criterion.

Message hooks verify exact reconstruction of actual HGT messages, normalized
attention, and unchanged model predictions. Checkpoint and parameter hashes
must remain unchanged. No full graph or final possession statistics are read.

Attention differences across separately trained models include learned Q/K
differences; they are not a pure causal effect of the controller. This audit
cannot by itself establish information redundancy, task-gradient dominance,
or insufficient predictive features.
