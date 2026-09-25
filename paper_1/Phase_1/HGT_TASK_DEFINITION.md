# Phase 1 HGT Task Definition

Scope: construct a Wyscout-based heterogeneous event graph and train an HGT model for next-event prediction. This document defines the Phase 1 modeling target only; it does not define Paper 2 robustness analysis.

## Objective

Use match event data to construct heterogeneous graphs and learn event-context representations with HGT. For each event step `t`, the model observes historical context up to event `t` and predicts key attributes of the next event `t+1`.

```text
Observed event context up to t
        ↓
Heterogeneous event graph
        ↓
HGT encoder
        ↓
current event/context representation
        ↓
multi-task prediction heads for event t+1
```

## Prediction Targets

For each target next event `e_{t+1}`, predict five groups of attributes.

### 1. Next Event Type

Predict the next event's main Wyscout event type.

Input label source:

```text
events[*].eventId
events[*].eventName
```

Current event classes:

```text
Duel
Foul
Free Kick
Goalkeeper leaving line
Interruption
Offside
Others on the ball
Pass
Save attempt
Shot
```

Task type:

```text
multi-class classification
```

### 2. Next Event Time Interval

Predict the time gap between current event `e_t` and next event `e_{t+1}`.

Input label source:

```text
events[*].matchPeriod
events[*].eventSec
```

Raw target:

```text
delta_t = absolute_time(e_{t+1}) - absolute_time(e_t)
```

Recommended initial task type:

```text
regression
```

Optional later version:

```text
bucketed classification
```

Notes:

```text
1H and 2H can be converted into absolute match seconds.
E1, E2, and P need explicit handling before training.
Events crossing matchPeriod boundaries need special conversion rules.
```

### 3. Next Event Position

Predict the next event's location.

Input label source:

```text
events[*].positions
```

Primary target:

```text
start_x, start_y
```

Optional secondary target:

```text
end_x, end_y
```

Recommended initial task type:

```text
coordinate regression
```

Optional later version:

```text
zone classification after spatial discretization
```

Notes:

```text
Most events have two position points.
Some events have only one position point.
For the first version, start position should be treated as mandatory and end position as optional or masked.
```

### 4. Executing Team and Player

Predict which team and which player will execute the next event.

Input label source:

```text
events[*].teamId
events[*].playerId
```

Task type:

```text
team prediction: multi-class classification within match teams
player prediction: multi-class classification over known players, with handling for playerId == 0
```

Notes:

```text
teamId is available and non-zero for all events.
playerId exists for all events, but playerId == 0 appears in part of the data.
The first implementation should either mask playerId == 0 for player prediction loss or assign it to an UNKNOWN_PLAYER class.
```

### 5. Favorable Event Direction

Predict whether the next event is favorable, unfavorable, or neutral for the executing team, using result tags.

Input label source:

```text
events[*].tags
analysis/tag_taxonomy.csv
analysis/event_tag_pipeline_overrides.csv
```

The label should be derived only from event-result tags, not from technical-action or event-context tags.

Initial label space:

```text
favorable
unfavorable
neutral_or_unknown
```

Candidate favorable signals:

```text
101 Goal
301 assist
302 keyPass
703 won
1401 interception
1801 accurate
```

Candidate unfavorable signals:

```text
102 own_goal
701 lost
1701 red_card
1702 yellow_card
1703 second_yellow_card
1802 not accurate
2001 dangerous_ball_lost
2101 blocked
```

Neutral or ignored signals:

```text
702 neutral
```

Important policy:

```text
Duel + neutral is explicitly marked in analysis/event_tag_pipeline_overrides.csv as no tag-edge construction.
For the favorable-direction label, neutral should not be treated as favorable or unfavorable.
```

Task type:

```text
multi-class classification
```

## Heterogeneous Graph Data Foundation

The graph should be built from the following existing data sources:

```text
raw/events/
raw/matches/
raw/entities/players.json
raw/entities/teams.json
raw/metadata/competitions.json
raw/mappings/eventid2name.csv
raw/mappings/tags2name.csv
analysis/tag_taxonomy.csv
analysis/event_tag_pipeline_overrides.csv
```

The damaged `raw/entities/referees.json` should not be used.

## Initial Node Types

Use only node types that are directly supported by current data and needed for Phase 1 prediction.

```text
Event
Player
Team
Match
Competition
EventType
Tag
TimeBin
Position
```

Notes:

```text
TimeBin and Position can be implemented as discretized nodes or as event attributes in the first version.
If using coordinate regression for next position, raw coordinates should remain event attributes.
```

## Initial Edge Types

Use typed edges that follow directly from the data.

```text
Event -> next -> Event
Player -> performs -> Event
Team -> performs -> Event
Event -> in_match -> Match
Match -> in_competition -> Competition
Event -> has_event_type -> EventType
Event -> has_tag -> Tag
Event -> by_team -> Team
Event -> by_player -> Player
```

Optional later edges:

```text
Event -> has_time_bin -> TimeBin
Event -> starts_at_position -> Position
Event -> ends_at_position -> Position
```

## Graph Unit

The first implementation should use match-level graphs.

```text
one match = one heterogeneous graph
```

Reason:

```text
The prediction target is next event within a match.
Match-level construction avoids leakage across future matches.
It also keeps the first HGT implementation easier to validate.
```

## Training Instance

Each training instance corresponds to one next-event prediction step.

```text
input: events up to e_t within the same match
target: attributes of e_{t+1}
```

The first implementation may use a fixed historical window for efficiency.

```text
input context: last K events before target
```

This keeps Phase 1 separate from Phase 2, where dynamic multi-scale subgraph selection will be introduced.

## Losses

The model should use a multi-task loss.

```text
L = L_event_type
  + L_delta_time
  + L_position
  + L_team
  + L_player
  + L_favorable_direction
```

Loss types:

```text
event type: cross entropy
delta time: MSE, MAE, or Huber
position: MSE, MAE, or Huber
team: cross entropy
player: cross entropy with UNKNOWN_PLAYER or masked zero-player events
favorable direction: cross entropy
```

## Required Preprocessing Decisions

Before implementation, decide the following:

```text
1. How to convert matchPeriod + eventSec into absolute time.
2. Whether playerId == 0 is masked or mapped to UNKNOWN_PLAYER.
3. Whether position prediction uses raw coordinates or zone classes in the first version.
4. Exact rule priority for favorable/unfavorable labels when one event has multiple result tags.
5. Whether Tag nodes include all tags or only tags allowed by pipeline overrides.
```

## Resolved Graph-Construction Decisions (Schema v1.1.0)

The first graph implementation resolves the preprocessing decisions as follows:

```text
1. Use a monotonic active-play clock. Period offsets preserve stoppage time and
   do not insert halftime breaks.
2. Represent playerId == 0 with UNKNOWN_PLAYER, but mask it from player loss.
3. Keep normalized raw start/end coordinates and position masks.
4. Resolve result-tag conflicts by event-specific priorities defined in
   analysis/event_result_direction_rules.csv. The model uses favorable versus
   unfavorable classification and masks neutral_or_unknown labels.
5. Keep all 59 Tag nodes for stable IDs, but create edges only when allowed by
   analysis/event_tag_pipeline_overrides.csv.
```

The graph schema, causal prefix requirement, output paths, and build commands
are documented in `GRAPH_CONSTRUCTION.md`.

## Non-Goals

This phase does not include:

```text
complex network robustness analysis
node or edge removal experiments
key-player definitions based on structural vulnerability
RAG explanation generation
dynamic multi-scale subgraph selection
```
