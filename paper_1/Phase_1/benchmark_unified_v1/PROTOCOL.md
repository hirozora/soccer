# Benchmark Protocol

## Dataset

The benchmark uses all 380 England 2017/18 matches from Wyscout Open Access and
the existing chronological match split:

| Split | Matches | Immediate transitions | Action-4 targets |
| --- | ---: | ---: | ---: |
| Train | 266 | 449,025 | 300,866 |
| Validation | 57 | 96,891 | 64,372 |
| Test | 57 | 96,854 | 64,939 |

The target is always the immediate next raw event. No event is removed and no
possession-end, half-end, or game-end event is inserted. Period changes only
disable the time loss. They do not remove the event or position target.

## Shared Action-4

| Label | Deterministic Open Wyscout rule |
| --- | --- |
| pass | Pass except Cross; Free Kick, Goal kick, Throw in; Clearance |
| dribble | Ground attacking duel with tag 501-504; Acceleration; untagged Touch |
| cross | Pass/Cross; Corner; Free kick cross |
| shot | Shot; Free kick shot; Penalty |

Events outside these four classes remain in every history. Their event loss and
event metrics are masked for the Seq2Event and NMSTPP contracts.

## Unified LEM

The fine label order is copied from the official 32-class tokenizer. Open
Wyscout fields are converted into V3-compatible booleans, after which the
official override order is applied: duel subtype and dribble, cross/long pass,
shot body part, save, yellow card, then red card.

The raw stream has 29 supported fine classes. `carry`, `first_half_end`, and
`game_end` are inactive and are masked before Softmax. Fine predictions are
folded with the training-only matrix

```text
M[raw,fine] = train_count(raw,fine) / train_count(fine)
```

This preserves observed ambiguity. For example, training-set `cross` maps to
Free Kick with probability 0.124 and Pass with probability 0.876. The complete
matrix is exported to `artifacts/unified_fold_matrix.csv`.

## Time And Position

Time is the active-play interval to the immediate next event, clipped to 60
seconds. Unified LEM predicts integer tokens 0-60 and is evaluated using the
probability-weighted expected second. HGT and NMSTPP predict a continuous value.

Coordinates use normalized Wyscout x/y. The NMSTPP comparison applies the 20
official Juego de Posicion centroids to both target and HGT prediction. Its
primary metrics are zone accuracy and Macro-F1; distance compares zone centres
for both models.

## Leakage Controls

- Player and team identity vocabularies are built from training matches only.
- Source events with no player use `UNK`; unseen validation/test players also use
  `UNK`, while their static metadata remains observable.
- Class weights and the fine-to-raw matrix use training targets only.
- Every sample is identified by `match_id:current_event_index`, and all paired
  prediction files must contain identical IDs and masks.

