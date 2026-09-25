# Phase 1 Heterogeneous Graph Construction

This directory contains the first reproducible graph-construction pipeline for
Paper 1. Raw Wyscout files and existing tag-analysis tables are treated as
immutable inputs.

## Output Unit

One match is serialized as one portable PyTorch tensor dictionary:

```text
data/whyscout/processed/heterogeneous_graphs/v1/
  graphs/<competition>/<match_id>.pt
  metadata/build_manifest.json
  metadata/graph_schema.json
  metadata/match_index.csv
  metadata/vocabularies.json
```

The representation is independent of `torch_geometric`. The `edge_stores`
dictionary already uses HGT-compatible typed `edge_index` tensors. Model code
can consume it directly or convert it to `HeteroData` after installing
`torch-geometric`.

## Node Types

```text
event
player
team
match
competition
event_type
tag
```

Time and coordinates remain event attributes in version 1. They are not turned
into nodes because the Phase 1 targets use continuous time and coordinate
regression.

Each match graph includes all ten event-type nodes and all 59 tag nodes so their
global indices are stable. Player nodes contain the match roster, event actors,
and an `UNKNOWN_PLAYER` node when `playerId == 0` occurs.

## Edge Types

```text
event -> next -> event
player -> performs -> event
event -> performed_by -> player
team -> performs -> event
event -> performed_by_team -> team
event -> in_match -> match
match -> contains -> event
match -> in_competition -> competition
competition -> contains -> match
event -> has_type -> event_type
event_type -> describes -> event
event -> has_tag -> tag
tag -> describes -> event
```

`Duel + neutral` follows
`analysis/event_tag_pipeline_overrides.csv`: the raw event remains present, but
the `event -> has_tag -> tag` and reverse tag edge are omitted for tag 702.

## Event Order and Time

Events are ordered by period, `eventSec`, and original source order as a final
tie-break. The period order is:

```text
1H -> 2H -> E1 -> E2 -> P
```

The absolute active-play clock does not insert halftime breaks. Each next-period
offset advances by the larger of the nominal period duration and the maximum
observed `eventSec` in the previous period. This preserves stoppage time and
prevents a 2H event near zero seconds from appearing earlier than the last 1H
event.

## Supervision Targets

Every event node except the final event carries labels shifted from the next
event:

```text
event_type_index
delta_seconds
start_position and start_position_mask
end_position and end_position_mask
team_local_index
player_vocab_index and player_known_mask
result_direction
```

Positions are normalized from `[0, 100]` to `[0, 1]`. `playerId == 0` and actor
IDs absent from `players.json` remain in the graph but are excluded from player
prediction loss through `player_known_mask`.

Result direction uses three classes:

```text
0 neutral_or_unknown
1 favorable
2 unfavorable
```

Only result tags define this target, but their meaning is conditioned on the
event type. For example, `Goal` is favorable for a Shot and unfavorable for a
Save attempt; `interception` is favorable for a recovery and unfavorable for a
Pass. The highest-priority matching rule in
`analysis/event_result_direction_rules.csv` determines the stored label.

The graph stores three values so every event remains aligned. The Version 1
model trains a binary favorable/unfavorable head and masks
`neutral_or_unknown` events.

## Causal Training Contract

Complete match graphs are storage artifacts, not direct HGT training inputs.
Bidirectional entity/tag relations allow a complete graph to carry future event
information through shared nodes. Before predicting event `t+1`, model input
must be sliced to events `0..t`:

```python
from football_hgt.dataset import load_match_graph, slice_event_prefix

graph = load_match_graph(".../graphs/World_Cup/2058017.pt")
prefix = slice_event_prefix(graph, end_event_index=99)
target = prefix["targets"]
```

The prefix keeps the current event's next-event label but removes the next event
node and every edge incident to future event nodes.

## Commands

Run tests:

```bash
cd /home/li/football/paper_1/Phase_1
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Build a small validation set:

```bash
python3 scripts/build_heterogeneous_graphs.py \
  --competitions World_Cup \
  --limit-matches 2 \
  --output-root data/whyscout/processed/heterogeneous_graphs/smoke
```

Build all competitions:

```bash
python3 scripts/build_heterogeneous_graphs.py
```

Validate every saved graph against the index and manifest:

```bash
python3 scripts/validate_heterogeneous_graphs.py
```

Existing graph files are reused unless `--overwrite` is passed. Every run
rewrites the metadata manifest and match index for the selected scope.

## Current Full Build

Schema `v1.1.0` uses event-conditioned result-direction rules from
`analysis/event_result_direction_rules.csv`. The graph structure is unchanged
from v1.0.0; only the Advantage label policy is revised.

The dataset has been built for all seven competitions:

```text
matches: 1,941
events: 3,251,294
next-event supervision steps: 3,249,353
retained event-tag edges: 4,399,292
graph dataset size: about 1.1G
```

Event-conditioned result labels before the final-event shift are:

```text
favorable: 1,957,561
unfavorable: 778,351
neutral_or_unknown: 515,382
```

`neutral_or_unknown` remains stored for alignment and is masked from the binary
Version 1 Advantage loss.

All 1,941 graph files passed independent reload and structural validation. The
machine-readable source of truth is
`data/whyscout/processed/heterogeneous_graphs/v1/metadata/validation_report.json`.
