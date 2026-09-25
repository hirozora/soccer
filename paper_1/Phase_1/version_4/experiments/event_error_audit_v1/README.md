# Original Event Head Error Audit

Validation only: 7,296 unique next-event samples from 57 England matches, evaluated by three fixed seeds.
Counts below are mean prediction counts across seeds, not 21,888 independent events.
Mean errors per seed: 1694.0. Two largest directional confusions account for 55.9% of errors.

| True next Event | Predicted as | Mean errors | Fraction of true class |
|---|---|---:|---:|
| Duel | Pass | 604.3 | 29.5% |
| Others on the ball | Pass | 343.3 | 60.4% |
| Pass | Duel | 198.3 | 5.4% |
| Pass | Interruption | 68.3 | 1.9% |
| Interruption | Pass | 65.0 | 19.9% |
| Others on the ball | Duel | 52.3 | 9.2% |
| Pass | Others on the ball | 51.3 | 1.4% |
| Shot | Pass | 36.7 | 40.7% |
| Foul | Pass | 34.3 | 41.9% |
| Pass | Foul | 28.3 | 0.8% |
| Others on the ball | Interruption | 26.3 | 4.6% |
| Duel | Interruption | 26.0 | 1.3% |

| Class | Unique samples | Precision | Recall | F1 | Top-3 recall |
|---|---:|---:|---:|---:|---:|
| Duel | 2049 | 0.834 | 0.673 | 0.745 | 99.1% |
| Foul | 82 | 0.490 | 0.520 | 0.497 | 78.9% |
| Free Kick | 431 | 0.973 | 1.000 | 0.986 | 100.0% |
| Goalkeeper leaving line | 18 | 0.765 | 0.519 | 0.617 | 57.4% |
| Interruption | 326 | 0.645 | 0.708 | 0.671 | 90.9% |
| Offside | 25 | 0.000 | 0.000 | 0.000 | 36.0% |
| Others on the ball | 568 | 0.618 | 0.235 | 0.340 | 91.5% |
| Pass | 3657 | 0.749 | 0.900 | 0.818 | 99.8% |
| Save attempt | 50 | 0.980 | 0.967 | 0.973 | 98.0% |
| Shot | 90 | 0.462 | 0.396 | 0.426 | 73.0% |

## Interpretation Boundaries

- Most raw errors are concentrated at the Duel/Others versus Pass boundary; Offside is never the top prediction.
- True-class-conditioned recall is diagnostic, not the deployment probability of that class in a state.
- Training transition probabilities are reported separately for the actual 34,048 training samples and full 449,025 training transitions.
- State and transition statistics are descriptive. They do not establish that a transition prior or hierarchical head will improve prediction.
- Target subevent information is audit-only, never an input feature. UNKNOWN means the stored graph has no subevent label.
- Source Event UIDs, adjacency targets and anchor states were checked against unchanged V3 graphs. No test predictions or test graphs were loaded.

## Artifacts

![Row-normalized confusion](confusion_heatmap.png)

See per_class.csv, top_confusions.csv, confusions_by_anchor_context.csv, training_transition_statistics.csv, class_support.csv and summary.json.
