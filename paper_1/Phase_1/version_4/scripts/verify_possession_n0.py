#!/usr/bin/env python
"""Verify that the Semantic V3 no-Possession path exactly reproduces V2."""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "benchmark_unified_v1/src")]

from football_benchmark.data import CanonicalEventDataset, collate_semantic_hgt, load_records  # noqa: E402
from football_benchmark.protocol import ProtocolArtifacts  # noqa: E402
from football_benchmark.sampling import TargetSamplePlan  # noqa: E402
from football_hgt_targets_v4.constants import (  # noqa: E402
    FEASIBILITY_ARTIFACT,
    POSSESSION_GRAPH_ROOT,
    SAMPLE_PLAN,
    SEMANTIC_GRAPH_ROOT,
    WINDOW_SIZE,
)
from football_hgt_targets_v4.model import build_target_model  # noqa: E402
from football_hgt_targets_v4.possession_data import collate_possession_hgt  # noqa: E402


def _batch(root: Path, artifacts, selected, collate, samples: int):
    dataset = CanonicalEventDataset(
        load_records("validation", graph_root=root),
        artifacts,
        WINDOW_SIZE,
        max_samples=samples,
        selected_currents=selected,
    )
    return next(iter(DataLoader(dataset, batch_size=samples, collate_fn=collate)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "experiments/loss_balance_validation/j2_020/seed20260715/best.pt",
    )
    args = parser.parse_args()
    artifacts = ProtocolArtifacts.load(FEASIBILITY_ARTIFACT)
    selected = TargetSamplePlan.load(SAMPLE_PLAN).currents_by_match("validation")
    v2 = _batch(
        SEMANTIC_GRAPH_ROOT,
        artifacts,
        selected,
        partial(collate_semantic_hgt, artifacts=artifacts, window_size=WINDOW_SIZE),
        args.samples,
    )
    v3 = _batch(
        POSSESSION_GRAPH_ROOT,
        artifacts,
        selected,
        partial(
            collate_possession_hgt,
            artifacts=artifacts,
            window_size=WINDOW_SIZE,
            topology="none",
            feature_level="topology",
            snapshot_scope="selected_events",
        ),
        args.samples,
    )
    methods = {"event": "ce", "time": "current_huber", "position": "xy"}
    model = build_target_model(
        artifacts,
        "joint",
        "j2_020",
        methods,
        graph_variant="semantic_v2",
        possession_topology="none",
        possession_feature_level="topology",
        dropout=0.0,
    )
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    with torch.no_grad():
        expected = model(v2)
        observed = model(v3)
    differences = {
        name: float((expected[name] - observed[name]).abs().max())
        for name in expected
    }
    edge_equal = set(v2["graph"].edge_index_dict) == set(v3["graph"].edge_index_dict) and all(
        torch.equal(v2["graph"].edge_index_dict[key], v3["graph"].edge_index_dict[key])
        for key in v2["graph"].edge_index_dict
    )
    passed = edge_equal and max(differences.values()) < 1e-6
    payload = {
        "passed": passed,
        "samples": args.samples,
        "checkpoint": str(args.checkpoint),
        "sample_ids_equal": v2["sample_ids"] == v3["sample_ids"],
        "edges_equal": edge_equal,
        "max_output_differences": differences,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not passed:
        raise SystemExit(1)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
