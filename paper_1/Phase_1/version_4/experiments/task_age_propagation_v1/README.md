# Task-conditioned Age Propagation Residual

This experiment compares fixed, shared-age, and task-age low-rank propagation
residuals on the frozen Semantic V3 experiment contract. Model selection uses
validation only; test results cannot change `selection/validation_lock.json`.

`AP-Task` test results were visible before this experiment was designed. Their
comparison with `PG-Task` is therefore a mechanism comparison, not a pristine
holdout comparison.
