# Position Head Refit V1

PR-Original reuses each Partial-L2-F80 best_guarded_core checkpoint.
PR-Refit warm-starts its Linear(64,2) Position Head (130 parameters) and
optimizes only those parameters on detached, eval-mode cached contexts.
The original model and all previous experiment artifacts remain read-only.

Training uses 34,048 fixed transitions, validation uses 7,296, with seeds
20260715-20260717. AdamW has a fresh optimizer, LR 3e-4, weight decay 1e-4,
batch 1024 and gradient clip 5. Raw XY Smooth L1 (beta=1) is averaged over
coordinates and all position-valid samples, without the joint-loss /3 factor.
There are at most 30 epochs, patience 5, and Epoch 0 participates in selection.
The selection tuple is (validation mean distance, position loss, epoch).

The primary population is position_mask, including unknown-player targets.
Match-cluster paired bootstrap uses 10,000 identical match draws for both
methods and all three model seeds. Each draw averages the per-seed,
sample-weighted improvements. No events or model seeds are independently
resampled. Validation requires >=0.25m improvement, two improving seeds,
a strictly positive lower 95% confidence bound, and unchanged outputs for
Event, Time, Team and Player (maximum error <1e-6).

Test reads require this experiment's passing validation lock. A failed
validation keeps Original and skips new test evaluation. A passing validation
locks the three best Heads before full-test evaluation. Historical test results
were already visible; this is a confirmation on an existing split, not a new
blind holdout. Previously trained MLP Null probes are background evidence only.

Run from version_4:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:../benchmark_unified_v1/src \
  python scripts/run_position_head_refit.py --stage pipeline --device cpu
```

Stages: verify, smoke, train, select, test, report, pipeline. Training supports
--seed and --resume-from. Interrupted runs resume from last.pt; a locked
experiment cannot be retrained. Sequential CPU jobs are the default because
each trainable Head is tiny. Online F80 verification uses native ATen instead
of this host's failing oneDNN GELU for irregular graph sizes; tolerances and
model definitions are unchanged.

The composite inference loader is position_head_refit.load_combined_model:
load the same source seed and an optional best_position.pt, replacing only
position_head. Report artifacts include per-seed metrics, mean/std, descriptive
Event/Zone/known-player groups, paired intervals and source artifact hashes.
