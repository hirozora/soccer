"""Separated data and model cost measurements for task receptive fields."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import torch
from football_benchmark.protocol import ProtocolArtifacts

from .constants import CONFIRMATION_SEEDS
from .fixed_budget_training import FixedBudgetConfig, _loader_config
from .five_task_training import _loader
from .model import build_partial_l2_model, build_receptive_field_model
from .receptive_field_study import ALL, RF_ROOT, checkpoint_path
from .training import _move_batch_to_device


def _configuration(name: str, device: str) -> FixedBudgetConfig:
    actual = "partial_l2" if name == "rf_f80" else name
    return FixedBudgetConfig(
        configuration=actual, output_dir=RF_ROOT / "efficiency", seed=CONFIRMATION_SEEDS[0],
        device=device, training_budget=24, batch_size=256, num_workers=2,
    )


@torch.inference_mode()
def benchmark_receptive_fields(device_name: str = "cuda:0", batches: int = 10) -> Path:
    device = torch.device(device_name)
    rows, branches = [], []
    for name in ALL:
        checkpoint = checkpoint_path(name, CONFIRMATION_SEEDS[0])
        if not checkpoint.exists():
            continue
        config = _configuration(name, device_name)
        artifacts = ProtocolArtifacts.load(config.artifact_path)
        model = (
            build_partial_l2_model(artifacts)
            if name == "rf_f80"
            else build_receptive_field_model(artifacts, name)
        ).to(device).eval()
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        loader = _loader("validation", _loader_config(config), artifacts, False)
        raw_batches = []
        started = time.perf_counter()
        iterator = iter(loader)
        for _ in range(min(batches, len(loader))):
            raw_batches.append(next(iterator))
        collate_seconds = time.perf_counter() - started

        transfer_seconds = 0.0
        for raw in raw_batches:
            tick = time.perf_counter()
            moved = _move_batch_to_device(raw, device)
            torch.cuda.synchronize(device)
            transfer_seconds += time.perf_counter() - tick
            del moved
        batch = _move_batch_to_device(raw_batches[0], device)
        for _ in range(3): model(batch)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        tick = time.perf_counter()
        for _ in range(10): model(batch)
        torch.cuda.synchronize(device)
        forward_seconds = time.perf_counter() - tick

        tick = time.perf_counter()
        sample_count = 0
        iterator = iter(_loader("validation", _loader_config(config), artifacts, False))
        for _ in range(min(batches, len(loader))):
            raw = next(iterator)
            sample_count += len(raw["sample_ids"])
            model(_move_batch_to_device(raw, device))
        torch.cuda.synchronize(device)
        full_seconds = time.perf_counter() - tick
        rows.append({
            "configuration": name,
            "samples": sample_count,
            "collate_ms_per_batch": 1000 * collate_seconds / len(raw_batches),
            "h2d_ms_per_batch": 1000 * transfer_seconds / len(raw_batches),
            "forward_ms_per_batch": 1000 * forward_seconds / 10,
            "full_inference_ms_per_batch": 1000 * full_seconds / len(raw_batches),
            "samples_per_second": sample_count / full_seconds,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        })
        graph = raw_batches[0]["graphs"]["f80"]
        branches.append({
            "configuration": name, "branch": "f80",
            "effective_events": int(graph["event"].num_nodes),
            "effective_nodes": sum(int(graph[node].num_nodes) for node in graph.node_types),
            "effective_edges": sum(int(graph[edge].edge_index.shape[1]) for edge in graph.edge_types),
            "shared_l1_calls": 1, "main_l2_calls": 1, "player_l2_calls": 1,
        })
        for count, metadata in raw_batches[0].get("rf_metadata", {}).items():
            branches.append({
                "configuration": name, "branch": f"n{count}",
                "effective_events": int(metadata["event_pool_mask"].sum()),
                "effective_nodes": sum(int(mask.sum()) for mask in metadata["node_masks"].values()),
                "effective_edges": sum(int(mask.sum()) for mask in metadata["edge_masks"].values()),
                "shared_l1_calls": 1, "main_l2_calls": 1, "player_l2_calls": 0,
            })
        del model, batch, raw_batches
        torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    baseline = frame[frame.configuration == "rf_f80"].iloc[0]
    for column in ("collate_ms_per_batch", "h2d_ms_per_batch", "forward_ms_per_batch", "full_inference_ms_per_batch", "peak_cuda_memory_bytes"):
        frame[f"{column}_relative_to_f80"] = frame[column] / baseline[column]
    output = RF_ROOT / "report"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "efficiency.csv", index=False)
    pd.DataFrame(branches).to_csv(output / "effective_graph_cost.csv", index=False)
    (output / "efficiency_metadata.json").write_text(json.dumps({
        "device": device_name, "timed_batches": batches,
        "contract": "One collated F80 graph; each unique logical RF performs its own two HGT layers.",
    }, indent=2), encoding="utf-8")
    return output
