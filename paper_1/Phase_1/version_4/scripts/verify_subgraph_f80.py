#!/usr/bin/env python
"""Verify the new F80 selector against saved pre-scale D2 predictions."""

from __future__ import annotations

import json
import math
import sys
from functools import partial
from pathlib import Path

import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "benchmark_unified_v1/src")]

from football_benchmark.data import CanonicalEventDataset, load_records  # noqa: E402
from football_benchmark.protocol import ProtocolArtifacts  # noqa: E402
from football_benchmark.sampling import TargetSamplePlan  # noqa: E402
from football_hgt_targets_v4.constants import (  # noqa: E402
    FEASIBILITY_ARTIFACT,
    POSSESSION_GRAPH_ROOT,
    SAMPLE_PLAN,
    WINDOW_SIZE,
)
from football_hgt_targets_v4.losses import compute_loss  # noqa: E402
from football_hgt_targets_v4.metrics import PredictionAccumulator  # noqa: E402
from football_hgt_targets_v4.model import build_target_model  # noqa: E402
from football_hgt_targets_v4.possession_data import collate_possession_hgt  # noqa: E402
from football_hgt_targets_v4.subgraph_study import (  # noqa: E402
    SUBGRAPH_EXPERIMENT_ROOT,
    f80_validation_dir,
)


def _saved_loss(frame: pd.DataFrame) -> float:
    rows = torch.arange(len(frame))
    target = torch.tensor(frame.event_true.to_numpy(), dtype=torch.long)
    probabilities = torch.tensor(
        frame[[column for column in frame if column.startswith("event_probability_")]].to_numpy(),
        dtype=torch.float32,
    )
    event = -torch.log(probabilities[rows, target].clamp_min(1e-12)).mean() / math.log(10)
    time_pred = torch.tensor(frame.time_pred.to_numpy(), dtype=torch.float32)
    time_true = torch.tensor(frame.time_true.to_numpy(), dtype=torch.float32)
    time_mask = torch.tensor(frame.time_mask.to_numpy(), dtype=torch.bool)
    time = F.smooth_l1_loss(
        time_pred / 60.0,
        time_true / 60.0,
        reduction="none",
        beta=1.0 / 60.0,
    )[time_mask].mean()
    position_pred = torch.tensor(frame[["position_pred_x", "position_pred_y"]].to_numpy(), dtype=torch.float32)
    position_true = torch.tensor(frame[["position_true_x", "position_true_y"]].to_numpy(), dtype=torch.float32)
    position_mask = torch.tensor(frame.position_mask.to_numpy(), dtype=torch.bool)
    position = F.smooth_l1_loss(position_pred, position_true, reduction="none").mean(-1)[position_mask].mean()
    return float((0.2 * event + time + position) / 3.0)


def main() -> None:
    seed = 20260715
    artifacts = ProtocolArtifacts.load(FEASIBILITY_ARTIFACT)
    selected = TargetSamplePlan.load(SAMPLE_PLAN).currents_by_match("validation")
    dataset = CanonicalEventDataset(
        load_records("validation", graph_root=POSSESSION_GRAPH_ROOT),
        artifacts,
        WINDOW_SIZE,
        max_samples=2,
        selected_currents=selected,
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=partial(
            collate_possession_hgt,
            artifacts=artifacts,
            window_size=WINDOW_SIZE,
            topology="membership",
            feature_level="dynamic",
            snapshot_scope="selected_events",
            context_view="f80",
        ),
    )
    batch = next(iter(loader))
    methods = {"event": "ce", "time": "current_huber", "position": "xy"}
    model = build_target_model(
        artifacts,
        "joint",
        "d2_dynamic",
        methods,
        graph_variant="semantic_v3_possession",
        possession_topology="membership",
        possession_feature_level="dynamic",
        dropout=0.1,
    )
    checkpoint = f80_validation_dir(seed) / "best.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    with torch.no_grad():
        predictions = model(batch)
        loss, _ = compute_loss(
            predictions,
            batch,
            "joint",
            "d2_dynamic",
            artifacts,
            methods,
            {"event": 0.2, "time": 1.0, "position": 1.0},
        )
    accumulator = PredictionAccumulator("joint")
    accumulator.update(predictions, batch, loss)
    observed = accumulator.frame().sort_values("sample_id").reset_index(drop=True)
    saved = pd.read_parquet(f80_validation_dir(seed) / "validation_predictions.parquet")
    saved = saved[saved.sample_id.isin(observed.sample_id)].sort_values("sample_id").reset_index(drop=True)
    if observed.sample_id.tolist() != saved.sample_id.tolist():
        raise RuntimeError("F80 regression sample IDs differ")
    float_columns = [
        column
        for column in observed
        if column.startswith("event_probability_")
        or column in {"time_pred", "position_pred_x", "position_pred_y"}
    ]
    output_diff = float((observed[float_columns] - saved[float_columns]).abs().to_numpy().max())
    loss_diff = abs(float(loss) - _saved_loss(saved))
    graph = batch["graph"]
    relative_diff = 0.0
    for start, stop in zip(graph["event"].ptr[:-1], graph["event"].ptr[1:]):
        length = int(stop - start)
        expected = (torch.arange(length, dtype=torch.float32) - float(length - 1)) / 79.0
        relative_diff = max(
            relative_diff,
            float(torch.max(torch.abs(graph["event"].relative_features[start:stop, 0] - expected))),
        )
    report = {
        "samples": len(observed),
        "prediction_max_abs_difference": output_diff,
        "loss_abs_difference": loss_diff,
        "relative_index_max_abs_difference": relative_diff,
        "threshold": 1e-6,
        "passed": output_diff < 1e-6 and loss_diff < 1e-6 and relative_diff < 1e-6,
        "source_checkpoint": str(checkpoint),
    }
    output = SUBGRAPH_EXPERIMENT_ROOT / "verification"
    output.mkdir(parents=True, exist_ok=True)
    (output / "f80_equivalence.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["passed"]:
        raise RuntimeError(f"F80 regression failed: {report}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
