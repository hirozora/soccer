#!/usr/bin/env python
"""Build train-only identities, class weights, and the fine-to-raw fold matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from football_benchmark.constants import (  # noqa: E402
    DEFAULT_ARTIFACT_PATH,
    FEASIBILITY_ARTIFACT_PATH,
    FEASIBILITY_SAMPLE_PLAN_PATH,
)
from football_benchmark.constants import (  # noqa: E402
    ACTION_NAMES,
    FINE_EVENT_NAMES,
    RAW_EVENT_NAMES,
)
from football_benchmark.protocol import build_protocol_artifacts  # noqa: E402
from football_benchmark.data import load_records  # noqa: E402
from football_benchmark.sampling import (  # noqa: E402
    FEASIBILITY_SAMPLE_SEED,
    FEASIBILITY_TARGETS_PER_MATCH,
    build_feasibility_sample_plan,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("full", "feasibility"), default="full")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sample-plan-output", type=Path)
    parser.add_argument("--targets-per-match", type=int, default=FEASIBILITY_TARGETS_PER_MATCH)
    parser.add_argument("--sampling-seed", type=int, default=FEASIBILITY_SAMPLE_SEED)
    args = parser.parse_args()
    output = args.output or (
        FEASIBILITY_ARTIFACT_PATH if args.profile == "feasibility" else DEFAULT_ARTIFACT_PATH
    )
    sample_plan = None
    if args.profile == "feasibility":
        plan_path = args.sample_plan_output or FEASIBILITY_SAMPLE_PLAN_PATH
        sample_plan = build_feasibility_sample_plan(
            {split: load_records(split) for split in ("train", "validation", "test")},
            targets_per_match=args.targets_per_match,
            sampling_seed=args.sampling_seed,
        )
        sample_plan.save(plan_path)
        rows = [
            {
                "split": split,
                "match_id": match_id,
                "current_event_index": current,
                "sample_id": f"{match_id}:{current}",
            }
            for split, matches in sample_plan.selections.items()
            for match_id, currents in matches.items()
            for current in currents
        ]
        pd.DataFrame(rows).to_csv(Path(plan_path).with_name("sample_ids.csv"), index=False)
    artifacts = build_protocol_artifacts(sample_plan=sample_plan)
    path = artifacts.save(output)
    summary = {
        **artifacts.metadata,
        "path": str(path),
        "players": len(artifacts.player_to_index),
        "teams": len(artifacts.team_to_index),
        "tags": len(artifacts.tag_to_index),
        "active_fine_classes": int(artifacts.fine_active_mask.sum()),
        "sample_plan": (
            None
            if sample_plan is None
            else {
                "path": str(args.sample_plan_output or FEASIBILITY_SAMPLE_PLAN_PATH),
                "sampling_seed": sample_plan.sampling_seed,
                "targets_per_match": sample_plan.targets_per_match,
                "split_samples": {
                    split: sample_plan.sample_count(split)
                    for split in ("train", "validation", "test")
                },
            }
        ),
        "class_counts": {
            name: values.tolist() for name, values in artifacts.class_counts.items()
        },
    }
    summary_path = path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(
        artifacts.fold_matrix.numpy(),
        index=RAW_EVENT_NAMES,
        columns=FINE_EVENT_NAMES,
    ).rename_axis("raw_event").to_csv(path.parent / "unified_fold_matrix.csv")
    weight_rows = []
    names_by_target = {
        "raw10": RAW_EVENT_NAMES,
        "action4": ACTION_NAMES,
        "fine32": FINE_EVENT_NAMES,
        "zone20": tuple(str(index) for index in range(20)),
    }
    for target, weights in artifacts.class_weights.items():
        for label, count, weight in zip(
            names_by_target[target], artifacts.class_counts[target], weights
        ):
            weight_rows.append(
                {
                    "target": target,
                    "label": label,
                    "count": int(count),
                    "weight": float(weight),
                }
            )
    pd.DataFrame(weight_rows).to_csv(path.parent / "class_weights.csv", index=False)
    label_manifest = {
        "raw10": list(RAW_EVENT_NAMES),
        "action4": list(ACTION_NAMES),
        "fine32": list(FINE_EVENT_NAMES),
        "fine32_active": {
            name: bool(artifacts.fine_active_mask[index])
            for index, name in enumerate(FINE_EVENT_NAMES)
        },
        "zone20": list(range(20)),
    }
    (path.parent / "label_manifest.json").write_text(
        json.dumps(label_manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
