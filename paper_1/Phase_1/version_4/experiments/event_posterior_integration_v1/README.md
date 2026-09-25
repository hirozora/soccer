# Event Posterior Integration V1

Frozen Partial-L2-F80 + locked Position Refit + TC-SoftPred Player output
(lambda 1.5). Event conditioning uses the *raw*, pre-TC Player posterior.

## Registered Models

- EI-Original: unchanged Event logits.
- EI-Null: original logits plus a residual with a zero 64D condition.
- EI-BasePost: original logits plus a residual conditioned on the raw
  posterior-weighted private Player state.

Residual: 128 -> 64 -> GELU -> Dropout(0.1) -> 10, 8,906 parameters.
Only the residual is trained; its final layer starts at zero. All original
parameters are frozen. Null and BasePost share initialization and RNG streams.

All 34,048 train / 7,296 validation events are used, including unknown Player.
Seeds 20260715-20260717; AdamW 3e-4, weight decay 1e-4, batch 1024,
clip 5, 30 epochs maximum, patience 5. Epoch zero is eligible. Select by
(-Macro-F1, -Accuracy, CE, epoch) subject to Accuracy >= Original - 0.01.

BasePost must beat both Original and Null by >= 0.005 mean Macro-F1,
with >= 2 positive seeds and a positive lower 95% CI, and lose <= 1pp
Accuracy. Bootstrap: 10,000 shared match draws, pooled confusion matrices
within each fixed seed, then arithmetic mean across seeds.

No test access before a passing validation lock. If validation fails, retain
Original and do not evaluate test. Existing test results were previously seen;
this is a confirmation on an existing split, not a pristine holdout.

## Execution

```bash
python scripts/run_event_posterior_integration.py --stage pipeline --device cpu
```

Stages: verify, smoke, train, select, test, report, pipeline. The train stage
accepts --seed, --mode, and --resume-from. A completed pipeline can be rerun
without retraining. Original artifacts are never overwritten.

## Numerical And Provenance Checks

Old Stage B cache metadata may self-reference its condition-cache path as
source_oracle_cache. We record this defect and independently hash the actual
Oracle checkpoint/cache, align all labels/IDs and recompute every expected
state. No existing cache is modified.

Original GPU-produced Event logits can differ by a few float32 ulps from CPU
GEMM. Cross-backend comparisons record raw logit error, require equivalent
probabilities (<1e-6) and exactly identical argmax. The same-backend zero-init
logit comparison and all four unaffected outputs retain the strict <1e-6
absolute criterion. Native ATen avoids this host's irregular-size oneDNN GELU
allocation failure; it does not alter the activation definition.
