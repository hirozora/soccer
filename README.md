# Soccer: Semantic HGT Football Event Prediction

Research code and result snapshot for England Wyscout next-event prediction.
The five targets are Event Type, Time Gap, Position, Team and Player.

## Research Sequence

1. Event heterogeneous graphs and causal possession inference (Semantic V2/V3).
2. Event losses, time targets, XY/Zone position targets and joint loss scaling.
3. Possession topology and causal state feature integration.
4. Temporal, possession, transition and spatial context scales; size-matched controls.
5. Task-specific hard routing and static P1/P2 context fusion.
6. Three/four/five-task conflict analysis and fixed-budget training controls.
7. Partial sharing: shared HGT Layer 1, separate Main/Player Layer 2.
8. Strict receptive fields, age-aware readout, age propagation residuals and
   continuous spatiotemporal edge gating.
9. Oracle task dependencies, player-centric history and player-posterior conditioning.
10. Position Head Refit, Team-aware Player scoring and Event confusion experiments.
11. Equal-step supervision coverage and causal cross-match history priors (completed).
12. Rotate Event confusion audit and Main-L2 Possession-transition ECA (completed; no upgrade).

## Layout

- `paper_1/Phase_1/src`, `scripts`, `tests`: graph and possession construction.
- `paper_1/Phase_1/benchmark_unified_v1`: shared datasets, batching and baselines.
- `paper_1/Phase_1/version_1` through `version_3`: earlier experiment code/results.
- `paper_1/Phase_1/version_4`: main controlled experiments, protocols and tests.
- `paper_1/Phase_1/version_4/experiments`: per-experiment reports, metrics and locks.
- `SNAPSHOT_MANIFEST.json`: exported file hashes and snapshot status.

The original relative directory structure is preserved for imports and artifact paths.
This is a curated snapshot, not a full copy of the live experiment workspace.

## Current Validated Model

Partial-L2-F80 uses a shared encoder and first HGT layer, a Main second layer for
Event/Time/Position/Team, and a private Player second layer. Two validated additions
are the 130-parameter Position Head Refit and TC-SoftPred Player prior (lambda=1.5).

Three seeds of the original Partial-L2 checkpoint and Position Refit are included;
each full backbone checkpoint is approximately 6.7 MB. They retain their original
checkpoint dictionaries; use the corresponding source loaders rather than assuming
that the file is a bare state dict. Only load pickle-based Torch files you trust.

The current baseline is the Rotate-trained Partial-L2-F80 with Position Refit and
TC-SoftPred, selected on full validation and confirmed on the existing test split.
Cross-match history priors did not qualify for an Event upgrade. The subsequent
Possession-transition ECA study also retained Rotate; no new ECA test evaluation
was performed. See [experiment status](EXPERIMENT_STATUS.md) for final metrics.

The 2026-09-26 incremental code/report update is available directly in this tree
and listed in `UPDATE_20260926.json`. The downloadable archive and its manifest
remain the older immutable snapshot; its model files are original Partial-L2,
not Rotate checkpoints.

## Installation and Data

Use Python 3.11 or newer and an appropriate PyTorch/PyG CUDA environment:

```bash
python -m pip install -e paper_1/Phase_1/benchmark_unified_v1
cd paper_1/Phase_1/version_4
python -m pip install -r requirements.txt
python -m pip install -e .
```

Raw Wyscout files and generated graph tensors are **not redistributed**. Obtain the
dataset under its applicable terms and follow `Phase_1/GRAPH_CONSTRUCTION.md` and
`Phase_1/data/whyscout/POSSESSION_DATA.md`. The package expects data beneath
`paper_1/Phase_1/data/whyscout/`. Split tables and small protocol artifacts are included.
Some shell launchers and historical metadata retain original machine paths; review
those before running on another machine. Tests requiring datasets, GPUs or historical
checkpoints cannot all run from this lightweight snapshot alone.

Large caches, per-sample predictions, most checkpoints, background logs and raw data
are excluded. Some result JSON files reference these excluded files; these are
historical provenance records, not promises that all resume inputs are bundled.
Reports over 2 MB are listed as exclusions in the manifest. Model training is not
started automatically by installation.

## Interpretation

Validation chooses configurations; test confirms the locked choice. Test results from
earlier experiments were already visible when later studies were designed, so these
are confirmations on an existing split, not repeated pristine blind tests. Oracle
probes are diagnostic and are not deployable models. See each experiment's protocol
and report for eligibility, actor guards, bootstrap and selection rules.

## Complete Download

This GitHub tree shows code, protocols and selected compact reports. The complete
3,912-file curated snapshot (including all exported result records and three-seed
model checkpoints) is stored in [the archive](archives/soccer-research-snapshot.tar.gz).
Extract it in a separate directory to reproduce the original layout:

```bash
mkdir full-snapshot
tar -xzf archives/soccer-research-snapshot.tar.gz -C full-snapshot
```

The archive is about 28 MB compressed (104 MB uncompressed).
`SNAPSHOT_MANIFEST.json` describes the files **inside that archive**, not only the
subset displayed directly in GitHub. Verify the archive with `ARCHIVE_SHA256SUMS`.
The snapshot is fixed; live experiments may have advanced since its creation.
