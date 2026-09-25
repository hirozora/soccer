# Task-specific Receptive Field V1

This experiment keeps one collated Semantic V3-D2 F80 graph and constrains
message propagation separately for each prediction task.

| Configuration | Event | Time | Position | Team | Player |
|---|---:|---:|---:|---:|---:|
| RF-F80 | 80 | 80 | 80 | 80 | 80 |
| RF-CoreU5 | 5 | 5 | 5 | 80 | 80 |
| RF-CoreU10 | 10 | 10 | 10 | 80 | 80 |
| RF-Task | 10 | 10 | 5 | 80 | 80 |

The N5/N10 relation tensors are filtered before HGT attention normalization.
Both HGT layers, entity closure, event pooling, and causal Possession snapshots
use the same receptive-field boundary. RF-F80 reuses the existing Partial-L2
F80 result; the other three configurations train for 24 complete epochs.

Validation selects the lowest core E/T/P loss among epochs satisfying both
Team/Player guard sets. A configuration with no eligible epoch is recorded as
ineligible. Test data are unavailable until the validation lock is written.

The detached pipeline is managed by the user service:

```text
football-hgt-v4-task-rf.service
```

Runtime state is written to `background/status.json`; per-job logs are stored
inside each smoke, validation, and test result directory.
