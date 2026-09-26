# Rotate Event Error Audit

Full validation only: 96,891 unique samples, 57 matches, three seeds.
Original and Rotate use exactly the same samples. No training or test access.
Counts are per-seed means; class recall is correctness conditional on the true class.

| Model | Accuracy | Macro-F1 | Errors |
|---|---:|---:|---:|
| original | 76.6397% | 0.6150 | 22634.0 |
| rotate | 77.9109% | 0.6360 | 21402.3 |

See per_class_mean.csv, confusions_mean.csv and anchor_groups_mean.csv.
Focus: Duel/Others -> Pass, reverse Pass -> Duel/Others, and anchor Pass + CONTROL.
The old 7,296-sample audit is not directly comparable by raw counts.
Error concentration does not establish a causal attention/hub failure.
