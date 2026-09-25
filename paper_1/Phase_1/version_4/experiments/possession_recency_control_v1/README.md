# Possession vs Equal-Size Recency Control

This experiment tests whether P1/P2 gains come from possession semantics or merely
from using fewer, more recent events.

## Registered Views

| Configuration | Input |
|---|---|
| `P1` | Current possession |
| `LP1` | Last `|P1|` same-period events |
| `P2` | Current and previous possession |
| `LP2` | Last `|P2|` same-period events |
| `Semantic SF-B` | Soft fusion of P1 and P2 |
| `Recency SF-B` | Soft fusion of LP1 and LP2 |

LP1 and LP2 match the corresponding possession view event count for every sample,
retain the same anchor and period, and contain no future event.

## Cost Contract

Single-view and fused costs are reported separately:

```text
Cost(Semantic SF-B) = Cost(P1) + Cost(P2)
Cost(Recency SF-B)  = Cost(LP1) + Cost(LP2)
```

The report includes measured Event/Node/Edge counts, GPU forward throughput,
end-to-end throughput, peak CUDA memory, and relative cost versus F80.

## Selection Discipline

All hypotheses and comparisons are locked from validation results in
`selection/validation_locked.json`. Test predictions cannot be generated before
that file exists and are used only to confirm the locked conclusions.
