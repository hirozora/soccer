# ECA Anchor/Age Attention Audit

Validation samples per seed: 96891; three frozen checkpoints per mode.
No training or test access. Pair age is destination Event local rank relative to anchor.
Mass sums next+gap attention after normalization over ALL incoming edges, averaged across four heads.
Primary averages give each sample equal weight within its age bucket. Pair-weighted means are supplementary.

| Anchor | Pair age | Base | Constant | Transition |
|---|---|---:|---:|---:|
| Pass+CONTROL | age_1_5 | 0.2065 | 0.1974 | 0.1519 |
| Pass+CONTROL | age_6_20 | 0.1930 | 0.1806 | 0.1410 |
| Pass+CONTROL | age_gt20 | 0.1836 | 0.1697 | 0.1334 |
| Pass+CONTROL | ends_at_anchor | 0.1809 | 0.1714 | 0.1154 |
| other | age_1_5 | 0.2033 | 0.1936 | 0.1613 |
| other | age_6_20 | 0.1953 | 0.1830 | 0.1453 |
| other | age_gt20 | 0.1879 | 0.1741 | 0.1383 |
| other | ends_at_anchor | 0.2580 | 0.2546 | 0.2343 |

CIs are descriptive paired match-cluster intervals (10000 shared draws).
Differences across separately trained models are not isolated causal effects of controller bias.
No change to the validation selection, test gate, or official Rotate baseline.
No-context/period-crossing pairs are excluded from the controlled-edge population; coverage is recorded per seed.
Attention does not measure information redundancy, task-gradient dominance, or feature sufficiency by itself.
