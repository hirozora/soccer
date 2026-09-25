"""Physical F80 integration checks for the cached Position-only experiment."""

from pathlib import Path

import torch

from football_benchmark.protocol import ProtocolArtifacts

from .constants import FEASIBILITY_ARTIFACT
from .five_task_training import FiveTaskTrainingConfig, _loader
from .oracle_dependency_study import source_checkpoint
from .position_head_refit import (
    ROOT, load_cache, load_combined_model, predict, run_dir, sha256, tensor_hash, write_json,
)
from .training import _move_batch_to_device


@torch.no_grad()
def verify_online(seed: int, root: Path = ROOT, head_checkpoint: Path | None = None,
                  device: str = "cpu") -> dict:
    cache, provenance = load_cache(seed, "validation", root)
    original = load_combined_model(seed, device=device)
    combined = load_combined_model(seed, head_checkpoint, device)
    common = lambda model: {k: v for k, v in model.state_dict().items() if not k.startswith("position_head.")}
    before = tensor_hash(common(original))
    if before != tensor_hash(common(combined)):
        raise RuntimeError("Replacing Position changed unrelated parameters")
    if any(p.requires_grad for p in combined.parameters()) or any(m.training for m in combined.modules()):
        raise RuntimeError("Combined inference must be frozen and in eval mode")
    config = FiveTaskTrainingConfig(
        mode="five_f80", output_dir=root, seed=seed, device=device,
        num_workers=0, max_validation_samples=32, batch_size=16,
    )
    loader = _loader("validation", config, ProtocolArtifacts.load(FEASIBILITY_ARTIFACT), False)
    row_for_id = {sample: row for row, sample in enumerate(cache["sample_ids"])}
    maximum = {key: 0.0 for key in ("event_logits", "time_seconds", "log_delta", "team_logits", "player_scores")}
    cache_error, original_error, samples = 0.0, 0.0, []
    call_counts = {"encoder": 0, "shared_l1": 0, "main_l2": 0, "player_l2": 0}

    def hook(name):
        def record(*_):
            call_counts[name] += 1
        return record

    handles = [combined.event_projection.register_forward_hook(hook("encoder")),
               combined.convolutions[0].register_forward_hook(hook("shared_l1")),
               combined.convolutions[1].register_forward_hook(hook("main_l2")),
               combined.player_convolution.register_forward_hook(hook("player_l2"))]
    batches = 0
    for raw in loader:
        rows = torch.tensor([row_for_id[sample] for sample in raw["sample_ids"]])
        for target_key, cached_key in (("position_xy", "position_true"), ("position_mask", "position_mask"),
                                       ("player_mask", "player_mask"), ("raw_event_10", "event_true")):
            if not torch.equal(raw["targets"][target_key].cpu(), cache[cached_key][rows]):
                raise RuntimeError(f"Physical targets differ: {target_key}")
        if not torch.equal(raw["current_event_indices"].cpu(), cache["current_event_indices"][rows]):
            raise RuntimeError("Physical anchors differ")
        batch = _move_batch_to_device(raw, torch.device(device))
        # This host's oneDNN GELU fails for irregular graph batch sizes;
        # native ATen retains the exact GELU definition without JIT allocation.
        with torch.backends.mkldnn.flags(enabled=False):
            old, new = original(batch), combined(batch)
        for key in maximum:
            maximum[key] = max(maximum[key], float((old[key] - new[key]).abs().max()))
        cached = predict(combined.position_head, cache["main_context"][rows])
        cache_error = max(cache_error, float((new["position_xy"].cpu() - cached).abs().max()))
        original_error = max(original_error, float((old["position_xy"].cpu() - cache["base_position_xy"][rows]).abs().max()))
        samples.extend(raw["sample_ids"])
        batches += 1
    for handle in handles:
        handle.remove()
    if not samples or any(count != batches for count in call_counts.values()):
        raise RuntimeError("Expected exactly one F80 encoding and each HGT branch per forward")
    if max([cache_error, original_error, *maximum.values()]) >= 1e-6:
        raise RuntimeError(f"Physical/cache equivalence failed: {maximum}, {cache_error}, {original_error}")
    if before != tensor_hash(common(combined)) or any(p.grad is not None for p in combined.parameters()):
        raise RuntimeError("Frozen model parameters or gradients changed")
    result = {"passed": True, "seed": seed, "device": device, "cpu_backend": "native_aten",
              "sample_ids": samples,
              "unaffected_output_max_errors": maximum, "online_cached_position_max_error": cache_error,
              "original_online_cached_position_max_error": original_error,
              "shared_parameter_hash": before, "layer_calls": call_counts,
              "source_checkpoint_sha256": sha256(source_checkpoint(seed)),
              "head_sha256": sha256(head_checkpoint) if head_checkpoint else None,
              "cache_provenance": provenance}
    path = run_dir(seed, root) / "online_verification.json" if head_checkpoint else root / "verification" / f"seed{seed}.json"
    write_json(path, result)
    return result
