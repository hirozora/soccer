# Causal Possession Data

This pipeline creates a versioned, graph-independent data source. It does not
claim to reconstruct an official Wyscout possession label.

## Pipeline

```text
Raw Wyscout
  -> processed/event_tables/v1/England
  -> processed/inferred_possessions/v1/England
```

Build both stages from `/home/li/football/paper_1/Phase_1`:

```bash
PYTHONPATH=src python scripts/build_wyscout_event_tables.py --competition England
PYTHONPATH=src python scripts/infer_wyscout_possessions.py --competition England
```

Pass `--overwrite` to rebuild an existing version.

## Causal Contract

The state machine processes events once from past to future. Control events can
confirm the acting team as owner. Duels, clearances, save attempts, and similar
contests can set a candidate but cannot confirm a switch. An opponent's later
control event starts a new possession at that event; prior rows are never
rewritten. Restarts always start a new possession. Boundaries and period changes
close the active state.

`event_possession_states.parquet` is the model-safe event-level table. Every
feature is a snapshot available immediately after that event. Columns in
`possessions.parquet` ending in `_audit_only` describe the eventual completed
record and must not be used as causal model inputs.

## Auditing

`metadata/validation_report.json` records structural and prefix-equivalence
checks. `analysis/` contains state, confidence, role, boundary, per-match, and
possession-length distributions plus a boundary-stratified manual audit sample.
The exact rules copied into the output are also available as
`possession_rules.csv`.

## Version 2

Version 2 preserves the complete V1 possession segmentation while correcting
context-dependent candidate direction, conflicting evidence, and period-first
state snapshots. Build it with PyArrow 24:

```bash
PYTHONPATH=src python scripts/infer_wyscout_possessions_v2.py --competition England
```

The output is stored in:

```text
processed/inferred_possessions/v2/England
```

The model-safe tables are `event_possession_states.parquet`,
`event_possession_evidence.parquet`, `possessions.parquet`, and
`possession_transitions.parquet`. `possession_audit.parquet` is forbidden as a
first-version model input.

For a graph anchored at event index `t`:

- read the open possession snapshot only from the event-state row at `t`;
- include a possession transition only when
  `transition_event_index <= t`;
- use `event_role` and `actor_relation_to_owner` as Event categorical features;
- encode `candidate_status` as `none`, `single`, or `conflicting`;
- never use a candidate team as a confirmed owner.

The V2 build is published only when its event-to-possession, owner, and switch
mapping is exactly identical to V1. The machine-readable contract and identity
report are in `metadata/model_safe_contract.json` and
`metadata/segmentation_identity.json`.
