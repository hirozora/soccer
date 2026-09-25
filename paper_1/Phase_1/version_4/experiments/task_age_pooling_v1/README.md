# Task-specific Age-aware Readout V1

This experiment keeps one Semantic V3-D2 F80 graph and one Partial-L2 HGT
forward. It compares the existing Event mean with a shared age-aware readout
and task-specific Event/Time/Position age-aware readouts.

```text
AP-Mean   : shared uniform Event mean (existing Partial-L2-F80)
AP-Shared : one learned age profile shared by Event/Time/Position
AP-Task   : three learned task-specific age profiles
```

Shared and task variants use identical age/scorer initialization and a common
initial pooling vector. The final scorer layer is zero initialized, so both
variants initially reproduce AP-Mean exactly. All choices are locked on
validation before test predictions are read.

Runtime state is written to `background/status.json`; job logs are stored in
their smoke, training, and test result directories.
