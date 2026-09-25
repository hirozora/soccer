#!/usr/bin/env python
"""Verify that the multi-view F80 path exactly reproduces Semantic V3-D2."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "benchmark_unified_v1/src")]

from football_benchmark.protocol import ProtocolArtifacts  # noqa: E402
from football_hgt_targets_v4.constants import (  # noqa: E402
    CONFIRMATION_SEEDS,
    FEASIBILITY_ARTIFACT,
    SAMPLE_PLAN,
)
from football_hgt_targets_v4.losses import compute_loss  # noqa: E402
from football_hgt_targets_v4.model import (  # noqa: E402
    build_multiview_model,
    build_target_model,
)
from football_hgt_targets_v4.multiview_study import MULTIVIEW_EXPERIMENT_ROOT  # noqa: E402
from football_hgt_targets_v4.subgraph_study import f80_validation_dir  # noqa: E402
from football_hgt_targets_v4.training import (  # noqa: E402
    TargetTrainingConfig,
    _loader,
)


def _config(mode: str | None) -> TargetTrainingConfig:
    return TargetTrainingConfig(
        task="joint",
        method=mode or "joint_optimized",
        joint_methods={"event": "ce", "time": "current_huber", "position": "xy"},
        joint_loss_weights={"event": 0.2, "time": 1.0, "position": 1.0},
        artifact_path=FEASIBILITY_ARTIFACT,
        sample_plan_path=SAMPLE_PLAN,
        output_dir=MULTIVIEW_EXPERIMENT_ROOT / "verification",
        learning_rate=9e-4,
        seed=CONFIRMATION_SEEDS[0],
        device="cpu",
        max_epochs=1,
        patience=1,
        effective_batch_size=2,
        micro_batch_size=2,
        num_workers=0,
        max_validation_samples=2,
        graph_variant="semantic_v3_possession",
        possession_topology="membership",
        possession_feature_level="dynamic",
        snapshot_scope="selected_events",
        context_view="f80",
        fusion_mode=mode,
    )


def main() -> None:
    artifacts = ProtocolArtifacts.load(FEASIBILITY_ARTIFACT)
    seed = CONFIRMATION_SEEDS[0]
    state = torch.load(f80_validation_dir(seed) / "best.pt", map_location="cpu", weights_only=False)
    standard = build_target_model(
        artifacts,
        "joint",
        "joint_optimized",
        {"event": "ce", "time": "current_huber", "position": "xy"},
        graph_variant="semantic_v3_possession",
        possession_topology="membership",
        possession_feature_level="dynamic",
    )
    multiview = build_multiview_model(artifacts, "f80")
    standard.load_state_dict(state["model"], strict=True)
    multiview.load_state_dict(state["model"], strict=True)
    standard.eval()
    multiview.eval()
    standard_batch = next(iter(_loader("validation", _config(None), artifacts, False)))
    multiview_batch = next(iter(_loader("validation", _config("f80"), artifacts, False)))
    if standard_batch["sample_ids"] != multiview_batch["sample_ids"]:
        raise RuntimeError("F80 verification sample IDs differ")
    with torch.no_grad():
        expected = standard(standard_batch)
        actual = multiview(multiview_batch)
        expected_loss, _ = compute_loss(
            expected,
            standard_batch,
            "joint",
            "joint_optimized",
            artifacts,
            _config(None).joint_methods,
            _config(None).joint_loss_weights,
        )
        actual_loss, _ = compute_loss(
            actual,
            multiview_batch,
            "joint",
            "f80",
            artifacts,
            _config("f80").joint_methods,
            _config("f80").joint_loss_weights,
        )
    prediction_difference = max(
        float((expected[name] - actual[name]).abs().max()) for name in expected
    )
    loss_difference = float((expected_loss - actual_loss).abs())
    threshold = 1e-6
    report = {
        "samples": len(standard_batch["sample_ids"]),
        "prediction_max_abs_difference": prediction_difference,
        "loss_abs_difference": loss_difference,
        "threshold": threshold,
        "passed": prediction_difference < threshold and loss_difference < threshold,
        "source_checkpoint": str(f80_validation_dir(seed) / "best.pt"),
    }
    output = MULTIVIEW_EXPERIMENT_ROOT / "verification"
    output.mkdir(parents=True, exist_ok=True)
    (output / "f80_equivalence.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
