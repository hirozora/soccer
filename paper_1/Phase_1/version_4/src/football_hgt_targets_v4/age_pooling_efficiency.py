"""Cost comparison for mean, age-aware, and strict-RF readouts."""

from __future__ import annotations

import json
import time

import pandas as pd
import torch
from football_benchmark.protocol import ProtocolArtifacts

from .age_pooling_study import AGE_POOL_ROOT, checkpoint_path
from .constants import CONFIRMATION_SEEDS
from .fixed_budget_training import FixedBudgetConfig, _loader_config
from .five_task_training import _loader
from .model import build_age_pooling_model, build_partial_l2_model, build_receptive_field_model
from .receptive_field_study import checkpoint_path as rf_checkpoint_path
from .training import _move_batch_to_device


CONFIGURATIONS = ("ap_mean", "ap_shared", "ap_task", "rf_task")


def _fixed_config(name: str, device: str) -> FixedBudgetConfig:
    actual = "partial_l2" if name == "ap_mean" else name
    return FixedBudgetConfig(
        configuration=actual, output_dir=AGE_POOL_ROOT / "efficiency",
        seed=CONFIRMATION_SEEDS[0], device=device, training_budget=24,
        batch_size=256, num_workers=2,
    )


@torch.inference_mode()
def benchmark_age_pooling(device_name: str = "cuda:0", batches: int = 10):
    device = torch.device(device_name)
    rows = []
    for name in CONFIGURATIONS:
        if name == "rf_task":
            checkpoint = rf_checkpoint_path(name, CONFIRMATION_SEEDS[0])
            config = FixedBudgetConfig(
                configuration=name, output_dir=AGE_POOL_ROOT / "efficiency",
                seed=CONFIRMATION_SEEDS[0], device=device_name,
                training_budget=24, batch_size=256, num_workers=2,
            )
        else:
            checkpoint = checkpoint_path(name, CONFIRMATION_SEEDS[0])
            config = _fixed_config(name, device_name)
        if not checkpoint.exists():
            continue
        artifacts = ProtocolArtifacts.load(config.artifact_path)
        model = (
            build_partial_l2_model(artifacts)
            if name == "ap_mean"
            else build_receptive_field_model(artifacts, "rf_task")
            if name == "rf_task"
            else build_age_pooling_model(artifacts, "shared" if name == "ap_shared" else "task")
        ).to(device).eval()
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
        loader = _loader("validation", _loader_config(config), artifacts, False)
        iterator = iter(loader)
        tick = time.perf_counter()
        raw_batches = [next(iterator) for _ in range(min(batches, len(loader)))]
        collate_seconds = time.perf_counter() - tick
        transfer_seconds = 0.0
        for raw in raw_batches:
            tick = time.perf_counter()
            moved = _move_batch_to_device(raw, device)
            torch.cuda.synchronize(device)
            transfer_seconds += time.perf_counter() - tick
            del moved
        batch = _move_batch_to_device(raw_batches[0], device)
        for _ in range(3):
            model(batch)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        tick = time.perf_counter()
        for _ in range(10):
            model(batch)
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
            "parameters": sum(value.numel() for value in model.parameters()),
            "collate_ms_per_batch": 1000 * collate_seconds / len(raw_batches),
            "h2d_ms_per_batch": 1000 * transfer_seconds / len(raw_batches),
            "forward_ms_per_batch": 1000 * forward_seconds / 10,
            "full_inference_ms_per_batch": 1000 * full_seconds / len(raw_batches),
            "samples_per_second": sample_count / full_seconds,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "encoder_calls": 1,
            "shared_l1_calls": 1 if name != "rf_task" else 3,
            "main_l2_calls": 1 if name != "rf_task" else 3,
            "player_l2_calls": 1,
        })
        del model, batch, raw_batches
        torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    baseline = frame[frame.configuration == "ap_mean"].iloc[0]
    for column in (
        "parameters", "collate_ms_per_batch", "h2d_ms_per_batch",
        "forward_ms_per_batch", "full_inference_ms_per_batch",
        "peak_cuda_memory_bytes",
    ):
        frame[f"{column}_relative_to_mean"] = frame[column] / baseline[column]
    output = AGE_POOL_ROOT / "report"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "efficiency.csv", index=False)
    (output / "efficiency_metadata.json").write_text(json.dumps({
        "device": device_name,
        "timed_batches": batches,
        "latency_target_relative_to_mean": 1.10,
        "memory_target_relative_to_mean": 1.05,
    }, indent=2), encoding="utf-8")
    return output
