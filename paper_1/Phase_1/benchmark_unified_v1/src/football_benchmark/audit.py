"""Full-split protocol audit without constructing model batches."""

from __future__ import annotations

from typing import Any

import torch

from .constants import ACTION_NAMES, FINE_EVENT_NAMES, RAW_EVENT_NAMES
from .data import load_records
from .mappings import action4_label, position_to_zone, unified_fine_label
from .protocol import ProtocolArtifacts, event_tag_sets
from .sampling import TargetSamplePlan


def audit_protocol(
    artifacts: ProtocolArtifacts,
    sample_plan: TargetSamplePlan | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"splits": {}}
    all_sample_ids: set[str] = set()
    for split in ("train", "validation", "test"):
        records = load_records(split)
        raw_counts = torch.zeros(len(RAW_EVENT_NAMES), dtype=torch.long)
        action_counts = torch.zeros(len(ACTION_NAMES), dtype=torch.long)
        fine_counts = torch.zeros(len(FINE_EVENT_NAMES), dtype=torch.long)
        zone_counts = torch.zeros(20, dtype=torch.long)
        samples = action_samples = time_samples = position_samples = 0
        source_unknown_player_events = 0
        unseen_player_events = 0
        unseen_team_events = 0
        for record in records:
            graph = torch.load(record.graph_path, map_location="cpu", weights_only=True)
            event = graph["node_stores"]["event"]
            player = graph["node_stores"]["player"]
            team = graph["node_stores"]["team"]
            tags = event_tag_sets(graph)
            if int(event["num_nodes"]) != record.num_events:
                raise ValueError(f"Event count mismatch for {record.match_id}")
            player_raw = player["raw_id"][event["player_local_index"]]
            team_raw = team["raw_id"][event["team_local_index"]]
            source_unknown_player_events += sum(
                int(value) <= 0 for value in player_raw.tolist()
            )
            unseen_player_events += sum(
                int(value) > 0 and int(value) not in artifacts.player_to_index
                for value in player_raw.tolist()
            )
            unseen_team_events += sum(
                int(value) > 0 and int(value) not in artifacts.team_to_index
                for value in team_raw.tolist()
            )
            currents = (
                range(record.num_events - 1)
                if sample_plan is None
                else sample_plan.currents_by_match(split).get(record.match_id)
            )
            if currents is None:
                raise ValueError(f"Sample plan is missing {split} match {record.match_id}")
            for current in currents:
                sample_id = f"{record.match_id}:{current}"
                if sample_id in all_sample_ids:
                    raise ValueError(f"Duplicate sample ID {sample_id}")
                all_sample_ids.add(sample_id)
                target = current + 1
                raw = int(event["event_type_index"][target])
                event_id = artifacts.event_type_ids[raw]
                subevent_id = artifacts.subevent_type_ids[
                    int(event["subevent_type_index"][target])
                ]
                tag_ids = tags[target]
                action, action_mask = action4_label(event_id, subevent_id, tag_ids)
                fine = unified_fine_label(event_id, subevent_id, tag_ids)
                raw_counts[raw] += 1
                fine_counts[fine] += 1
                if action_mask:
                    action_counts[action] += 1
                    action_samples += 1
                if bool(event["start_position_mask"][target]):
                    position = event["start_position"][target]
                    if not torch.isfinite(position).all() or not bool(
                        ((position >= 0) & (position <= 1)).all()
                    ):
                        raise ValueError(f"Invalid position in {sample_id}")
                    zone_counts[int(position_to_zone(position))] += 1
                    position_samples += 1
                delta = float(graph["targets"]["delta_seconds"][current])
                if not 0 <= delta:
                    raise ValueError(f"Negative interval in {sample_id}")
                if int(event["period_index"][current]) == int(event["period_index"][target]):
                    time_samples += 1
                samples += 1
        result["splits"][split] = {
            "matches": len(records),
            "samples": samples,
            "action4_samples": action_samples,
            "time_samples": time_samples,
            "position_samples": position_samples,
            "source_unknown_player_events": source_unknown_player_events,
            "unseen_player_events": unseen_player_events,
            "unseen_team_events": unseen_team_events,
            "raw10_counts": raw_counts.tolist(),
            "action4_counts": action_counts.tolist(),
            "fine32_counts": fine_counts.tolist(),
            "zone20_counts": zone_counts.tolist(),
            "sampling_mode": (
                "full" if sample_plan is None else sample_plan.split_modes[split]
            ),
        }
    active_columns = artifacts.fine_active_mask
    column_sums = artifacts.fold_matrix[:, active_columns].sum(dim=0)
    if not torch.allclose(column_sums, torch.ones_like(column_sums), atol=1e-7):
        raise ValueError("Active Unified fold columns do not sum to one")
    if artifacts.fold_matrix[:, ~active_columns].abs().sum() != 0:
        raise ValueError("Inactive Unified fold columns must be zero")
    train_report = result["splits"]["train"]
    audit_counts = {
        "raw10": train_report["raw10_counts"],
        "action4": train_report["action4_counts"],
        "fine32": train_report["fine32_counts"],
        "zone20": train_report["zone20_counts"],
    }
    for target, values in audit_counts.items():
        if values != artifacts.class_counts[target].tolist():
            raise ValueError(f"Artifact and audited {target} counts differ")
    result["total_unique_samples"] = len(all_sample_ids)
    result["active_fine_classes"] = int(active_columns.sum())
    result["sampling_seed"] = None if sample_plan is None else sample_plan.sampling_seed
    result["status"] = "passed"
    return result
