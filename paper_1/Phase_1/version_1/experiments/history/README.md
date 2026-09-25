# Preliminary Experiment Archive

This directory contains all Version 1 experiments produced before 2026-07-22.
They are retained for traceability but should not be treated as the final reliable
baseline evaluation.

The archive includes:

```text
baseline_comparison/
baseline_comparison_equal_window_k80/
baseline_comparison_equal_window_k80_lr_*/
baseline_comparison_multi_seed/
window_ablation/
window_ablation_stage2/
window_k80_multiseed/
baseline_comparison*_summary.json
baseline_comparison_multi_seed_summary.csv
window_ablation_summary_v1.json
```

New experiments must write outside `history/` under the parent `experiments/`
directory and use a newly documented reliable protocol. Historical JSON files
are preserved unchanged, so paths embedded inside them may refer to their
original pre-archive locations.
