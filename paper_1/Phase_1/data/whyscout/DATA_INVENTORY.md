# Wyscout Data Inventory

Scope: raw data inspection and file organization only. This document does not define any future heterogeneous graph schema.

## Directory Layout

```text
analysis/
  event_result_direction_rules.csv
  event_tag_occurrence_by_event.csv
  event_tag_occurrence_with_taxonomy.csv
  event_tag_pipeline_overrides.csv
  tag_subcategory_annotation.csv
  tag_taxonomy.csv
  possession_rules.csv
raw/
  entities/
    coaches.json
    players.json
    referees.json
    teams.json
  events/
    events_England.json
    events_European_Championship.json
    events_France.json
    events_Germany.json
    events_Italy.json
    events_Spain.json
    events_World_Cup.json
  mappings/
    eventid2name.csv
    tags2name.csv
  matches/
    matches_England.json
    matches_European_Championship.json
    matches_France.json
    matches_Germany.json
    matches_Italy.json
    matches_Spain.json
    matches_World_Cup.json
  metadata/
    competitions.json
  ratings/
    playerank.json
processed/
  event_tables/
    v1/England/
      events.parquet
      event_tags.parquet
      matches.parquet
      match_teams.parquet
      metadata/
      analysis/
  inferred_possessions/
    v1/England/
      event_possession_states.parquet
      possessions.parquet
      possession_rules.csv
      metadata/
      analysis/
  heterogeneous_graphs/
    v1/
      graphs/<competition>/<match_id>.pt
      metadata/build_manifest.json
      metadata/graph_schema.json
      metadata/match_index.csv
      metadata/validation_report.json
      metadata/vocabularies.json
```

Total raw directory size: about 929M.

`processed/heterogeneous_graphs/v1/` is a derived Phase 1 dataset. Its graph
schema and regeneration commands are documented in
`/home/li/football/paper_1/Phase_1/GRAPH_CONSTRUCTION.md`.

`processed/event_tables/v1/England/` is a graph-independent normalization of
the raw England event stream. `processed/inferred_possessions/v1/England/`
contains deterministic, causal possession states inferred from that stream.
These tables do not modify either heterogeneous graph dataset. Their contract
and regeneration commands are documented in `POSSESSION_DATA.md`.

## File Counts

### Events

| file | records | matches | teams | players |
| --- | ---: | ---: | ---: | ---: |
| raw/events/events_England.json | 643150 | 380 | 20 | 515 |
| raw/events/events_European_Championship.json | 78140 | 51 | 24 | 452 |
| raw/events/events_France.json | 632807 | 380 | 20 | 542 |
| raw/events/events_Germany.json | 519407 | 306 | 18 | 474 |
| raw/events/events_Italy.json | 647372 | 380 | 20 | 534 |
| raw/events/events_Spain.json | 628659 | 380 | 20 | 558 |
| raw/events/events_World_Cup.json | 101759 | 64 | 32 | 602 |

Total event records: 3251294.

### Matches

| file | records | competitionId | status |
| --- | ---: | ---: | --- |
| raw/matches/matches_England.json | 380 | 364 | Played |
| raw/matches/matches_European_Championship.json | 51 | 102 | Played |
| raw/matches/matches_France.json | 380 | 412 | Played |
| raw/matches/matches_Germany.json | 306 | 426 | Played |
| raw/matches/matches_Italy.json | 380 | 524 | Played |
| raw/matches/matches_Spain.json | 380 | 795 | Played |
| raw/matches/matches_World_Cup.json | 64 | 28 | Played |

Total match records: 1941.

### Other JSON Files

| file | records |
| --- | ---: |
| raw/entities/coaches.json | 208 |
| raw/entities/players.json | 3603 |
| raw/entities/teams.json | 142 |
| raw/metadata/competitions.json | 7 |
| raw/ratings/playerank.json | 46897 |

`raw/entities/referees.json` is not valid JSON because it is truncated at the end. It contains 627 `wyId` occurrences before the parse failure.

### CSV Files

| file | rows | columns |
| --- | ---: | --- |
| raw/mappings/eventid2name.csv | 37 | event, subevent, event_label, subevent_label |
| raw/mappings/tags2name.csv | 60 | Tag, Label, Description |

## Top-Level Fields

### Event Files

All event files use the same top-level fields:

```text
eventId
eventName
eventSec
id
matchId
matchPeriod
playerId
positions
subEventId
subEventName
tags
teamId
```

Nested fields:

```text
positions[]: x, y
tags[]: id
```

Observed event periods:

```text
1H, 2H, E1, E2, P
```

Observed event names and counts across all event files:

| eventName | records |
| --- | ---: |
| Pass | 1665508 |
| Duel | 879083 |
| Others on the ball | 257240 |
| Free Kick | 193273 |
| Interruption | 130097 |
| Foul | 51049 |
| Shot | 43078 |
| Save attempt | 17619 |
| Offside | 8182 |
| Goalkeeper leaving line | 6165 |

Additional event data observations:

```text
positions length 1: 741 records
positions length 2: 3250553 records
playerId == 0: 226038 records
```

### Match Files

Top-level fields:

```text
competitionId
date
dateutc
duration
gameweek
groupName
label
referees
roundId
seasonId
status
teamsData
venue
winner
wyId
```

`groupName` appears only in 115 cup-tournament match records.

Observed duration values:

```text
Regular: 1931
ExtraTime: 3
Penalties: 7
```

Nested fields:

```text
referees[]: refereeId, role
teamsData[teamId]: coachId, formation, hasFormation, score, scoreET, scoreHT, scoreP, side, teamId
teamsData[teamId].formation: bench, lineup, substitutions
lineup[] / bench[]: assists, goals, ownGoals, playerId, redCards, yellowCards
substitutions[]: assists, minute, playerIn, playerOut
```

Observed referee roles:

```text
referee
firstAssistant
secondAssistant
fourthOfficial
firstAdditionalAssistant
secondAdditionalAssistant
```

### Players

Fields:

```text
birthArea
birthDate
currentNationalTeamId
currentTeamId
firstName
foot
height
lastName
middleName
passportArea
role
shortName
weight
wyId
```

Nested fields:

```text
birthArea: alpha2code, alpha3code, id, name
passportArea: alpha2code, alpha3code, id, name
role: code2, code3, name
```

Player role counts:

```text
Midfielder: 1257
Defender: 1200
Forward: 720
Goalkeeper: 426
```

### Teams

Fields:

```text
area
city
name
officialName
type
wyId
```

Nested fields:

```text
area: alpha2code, alpha3code, id, name
```

Team type counts:

```text
club: 98
national: 44
```

### Coaches

Fields:

```text
birthArea
birthDate
currentTeamId
firstName
lastName
middleName
passportArea
shortName
wyId
```

Nested fields:

```text
birthArea: alpha2code, alpha3code, id, name
passportArea: alpha2code, alpha3code, id, name
```

### Competitions

Fields:

```text
area
format
name
type
wyId
```

Nested fields:

```text
area: alpha2code, alpha3code, id, name
```

### PlayerRank

Fields:

```text
goalScored
matchId
minutesPlayed
playerId
playerankScore
roleCluster
```

Coverage:

```text
records: 46897
matches: 1941
players: 2719
```

## Data Quality Notes

1. `raw/entities/referees.json` is truncated and cannot be parsed as JSON.
2. Event files are valid JSON and share the same top-level schema.
3. Match files are valid JSON; `groupName` is optional and appears only in cup competitions.
4. `playerId == 0` appears in 226038 event records.
5. Most event records have two position objects; 741 records have only one.

## Derived Analysis Tables

| file | purpose |
| --- | --- |
| analysis/event_tag_occurrence_by_event.csv | Empirical event-tag co-occurrence counts by event type. |
| analysis/tag_taxonomy.csv | Manual tag taxonomy with only three pipeline-level categories: technical_action, event_context, event_result. |
| analysis/tag_subcategory_annotation.csv | Optional manual subcategory notes for human reference only; not intended as downstream pipeline input. |
| analysis/event_tag_occurrence_with_taxonomy.csv | Event-tag co-occurrence table joined only with the three-category tag taxonomy. |
| analysis/event_tag_pipeline_overrides.csv | Event-tag level downstream policy exceptions; currently marks `Duel` + `neutral` as no tag-edge construction. |
| analysis/event_result_direction_rules.csv | Event-conditioned favorable/unfavorable/neutral rules and conflict priorities for all 47 observed event-result tag pairs. |
