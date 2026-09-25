# Semantic V3 Possession Graphs

Semantic V3 preserves every Semantic V2 node, edge, Event tensor, and target,
then adds causal possession topology:

```text
Event <-> Possession <-> Team
Possession -> next -> Possession
```

The full-match graphs are stored at:

```text
../data/whyscout/processed/heterogeneous_graphs/semantic_v3_possession
```

## Build

The Possession V2 Parquet files require PyArrow 24 or 25. Install the isolated
builder dependency and run:

```bash
python -m pip install -e '.[possession-graph]'
python scripts/build_possession_graphs.py --competition England
```

The builder reads only the model-safe state, possession, and transition tables.
It never reads `possession_audit.parquet`. A complete build is staged in a
temporary directory and published only after all match and dataset checks pass.

## Causal Views

Use the explicit extraction interface:

```python
from football_benchmark.possession_graph import extract_causal_subgraph

subgraph = extract_causal_subgraph(
    graph,
    event_indices=[10, 14, 18],
    anchor_event_index=18,
    snapshot_scope="selected_events",
    include_dynamic_possession_features=False,
)
```

`anchor_history` exposes a causal possession snapshot built from all history up
to the anchor. `selected_events` retains the causally inferred possession owner
and topology but derives visible state only from selected Events. Dynamic fields
can be disabled without changing their shapes; their masks are then false and
their values are zero.

Reference-only IDs and local indices are listed separately from categorical,
continuous, and binary model fields in `metadata/graph_schema.json`. They must
not be treated as ordinary numeric model features.

The first topology experiment should use:

```text
snapshot_scope = selected_events
include_dynamic_possession_features = False
```

Transition reasons remain edge attributes. Cross-period transitions also carry
an independent `cross_period` boolean because the source state machine may keep
an earlier close reason while the transition itself crosses a period boundary.
