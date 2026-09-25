# Unified England Next-Event Benchmark

This package compares HGT, Soccer-Seq2Event, Unified LEM, and NMSTPP on the
same England 2017/18 Wyscout matches and immediate raw-event transitions.

The benchmark has three pair-specific contracts:

| Contract | Event target | Time target | Position target |
| --- | --- | --- | --- |
| `seq2event` | common physical action-4 | not predicted | continuous x/y |
| `unified_lem` | HGT raw-10 vs LEM fine-32 folded to raw-10 | clipped seconds | continuous x/y |
| `nmstpp` | common physical action-4 | clipped seconds | HGT x/y mapped to official zone-20 vs NMSTPP zone-20 |

All histories retain the full raw event stream. Unsupported action targets are
masked rather than removed, and no synthetic possession or period-end events
are inserted.

## Reproducible workflow

```bash
cd /home/li/football/paper_1/Phase_1/benchmark_unified_v1
python scripts/build_protocol.py
pytest
python scripts/run_experiment.py --contract seq2event --model seq2event \
  --window-size 80 --learning-rate 0.01 --seed 20260715 --smoke
python scripts/run_matrix.py --stage print
```

`run_matrix.py --stage print` prints the complete tuning and five-seed command
matrix without starting expensive training. Use `--stage tune` or
`--stage final` to execute the corresponding runs.

## Lightweight feasibility profile

The controlled feasibility profile keeps all 380 matches and every K=80 input
history, but samples target transitions by time within each match:

| Split | Targets per match | Samples | Use |
| --- | ---: | ---: | --- |
| Train | 128 | 34,048 | optimization and train-only statistics |
| Validation | 128 | 7,296 | LR and checkpoint selection |
| Test | all | 96,854 | final evaluation only |

The fixed sampling seed is `20260701`; it is independent of model training
seeds. All six model/contract endpoints load the same immutable sample plan.
The profile uses K=80, at most 8 epochs, patience 2, three training seeds, and
the same three-candidate LR budget for each model family. Seed `20260715` is
reused from tuning, resulting in 30 unique training runs rather than 78.

```bash
python scripts/build_protocol.py --profile feasibility
python scripts/audit_protocol.py \
  --artifact-path artifacts/feasibility/protocol.pt \
  --sample-plan-path artifacts/feasibility/sample_plan.json \
  --output artifacts/feasibility/audit.json
python scripts/run_feasibility_matrix.py --stage print
python scripts/run_feasibility_matrix.py --stage tune \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 --workers-per-device 3
python scripts/run_feasibility_matrix.py --stage final \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 --workers-per-device 3
python scripts/summarize.py --profile feasibility --replicates 10000
```

To run the entire pipeline detached from the terminal:

```bash
bash scripts/launch_feasibility_background.sh
cat experiments/feasibility/background/status.json
tail -f experiments/feasibility/background/pipeline.log
```

HGT uses a micro-batch of 256, equal to the fixed effective batch size, and each
job uses two data-loader workers. The launcher runs three independent jobs per
GPU by default. This improves throughput for these small models without
changing sample IDs, losses, optimizer-step batch size, or tuning budgets.

This profile is intended to establish architectural feasibility. It preserves
paired data, target definitions, tuning budgets, and full-test evaluation, but
three seeds and sampled optimization targets provide less statistical power
than the full protocol.

## Unified LEM repair

The original feasibility run exposed a fine-32 loss-scaling defect that caused
Unified LEM's event objective to be almost ignored. The isolated repair and its
results are documented in `UNIFIED_LEM_REPAIR.md`. The repaired combined report
is under `experiments/feasibility/summary_repaired`; it reuses every existing
non-Unified result rather than retraining those models.

## Semantic spatiotemporal HGT V2

`SEMANTIC_HGT_V2.md` documents the isolated bidirectional event-entity graph,
causal time-gap relations, Zone20 structure, relative window features, and the
HGT-only replacement experiment. Existing graph data and baseline runs remain
unchanged; the combined output is `experiments/feasibility/summary_semantic_v2`.
