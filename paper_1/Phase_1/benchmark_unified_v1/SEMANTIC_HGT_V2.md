# Semantic Spatiotemporal HGT V2

This experiment replaces only the HGT side of the controlled feasibility
benchmark. Unified LEM uses its repaired results; Soccer-SEQ2Event and NMSTPP
use their existing results.

## Graph

- Source: the immutable v1 England match graphs.
- Output: `data/whyscout/processed/heterogeneous_graphs/semantic_v2`.
- Nodes: Event, Player, Team, EventType, Tag, and the official 20 Zone nodes.
- Relations: causal next, exclusive historical time-gap buckets, and
  bidirectional event/entity relations.
- Window-only features: relative event index, time age, and current-team-view
  spatial displacement from each historical event to the anchor.

The full-match graph is storage only. Every training sample filters all event
endpoints to its causal K=80 prefix before HGT message passing.

## Model and experiment

Semantic HGT uses two 64-dimensional, four-head layers. Every node type is
updated with a residual and LayerNorm. Prediction still pools only Event nodes.

The matrix tunes three HGT contracts over `1e-4`, `3e-4`, and `9e-4`, then runs
seeds `20260715` through `20260717`. The detached launcher waits for all four
GPUs to become idle, measures peak memory with smoke runs, and caps concurrency
at three jobs per GPU.

```bash
bash scripts/launch_semantic_hgt_background.sh
cat experiments/feasibility/semantic_hgt_v2/background/status.json
tail -f experiments/feasibility/semantic_hgt_v2/background/pipeline.log
```

The final combined report is written to
`experiments/feasibility/summary_semantic_v2`. It contains paired bootstrap
results, legacy-versus-semantic HGT deltas, retained baseline hashes, and
relation-family removal diagnostics.
