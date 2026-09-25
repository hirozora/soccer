# State-aware Relation Gating V1

This experiment compares the locked `Partial-L2-F80` baseline (`G0`) with one
sample-state-conditioned relation-gating model (`G1`). Graph structure, F80
context, five target heads, losses, and Partial-L2 sharing remain unchanged.

For sample `b`, concrete relation `r`, and HGT layer `l`, the scalar gate is:

```text
g[b,r,l] = 2 * sigmoid(MLP_l([anchor_state[b], relation_embedding[r]]))
```

The same scalar is broadcast to every edge of that sample/relation/layer after
attention and before aggregation. The gate never reads a target-node state.
Its eight causal inputs are CONTROL, CONTESTED, restart, boundary, confirmed
switch, normalized possession duration, normalized event count, and snapshot
validity.

Execution is validation-gated:

```text
tests -> CUDA smoke -> 3-seed validation -> validation lock
      -> G1 test only if selected -> report
```

The test split cannot change the validation-selected model. If G1 fails the
validation criteria, G0 remains locked and no G1 test predictions are created.

Run in the background with:

```bash
scripts/launch_state_gating_background.sh
```
