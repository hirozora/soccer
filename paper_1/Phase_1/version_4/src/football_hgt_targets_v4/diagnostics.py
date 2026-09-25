"""Per-task gradient diagnostics on the shared Semantic HGT backbone."""

from __future__ import annotations

import math
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch

from football_benchmark.data import move_batch_to_device
from football_benchmark.protocol import ProtocolArtifacts

from .losses import compute_loss
from .model import TargetStudyHGT, build_target_model


HEAD_PREFIXES = (
    "event_head.",
    "time_head.",
    "position_head.",
    "time_bucket_head.",
    "time_offset_head.",
    "zone_head.",
    "zone_residual_head.",
)


def _shared_parameters(model: TargetStudyHGT) -> list[torch.nn.Parameter]:
    return [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not name.startswith(HEAD_PREFIXES)
        and not name.startswith("fusion_logits.")
    ]


def _dot(
    left: tuple[torch.Tensor | None, ...],
    right: tuple[torch.Tensor | None, ...],
) -> torch.Tensor:
    values = [
        (a * b).sum()
        for a, b in zip(left, right)
        if a is not None and b is not None
    ]
    if not values:
        raise RuntimeError("No shared gradients were produced")
    return torch.stack(values).sum()


def compute_gradient_diagnostics(
    model: TargetStudyHGT,
    batch: dict[str, Any],
    artifacts: ProtocolArtifacts,
    method: str,
    joint_methods: dict[str, str],
    joint_loss_weights: dict[str, float],
) -> dict[str, Any]:
    """Measure raw and effectively weighted gradients without updating the model."""

    was_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)
    predictions = model(batch)
    _, components = compute_loss(
        predictions,
        batch,
        "joint",
        method,
        artifacts,
        joint_methods,
        joint_loss_weights,
    )
    parameters = _shared_parameters(model)
    gradients: dict[str, tuple[torch.Tensor | None, ...]] = {}
    for index, name in enumerate(("event", "time", "position")):
        gradients[name] = torch.autograd.grad(
            components[name],
            parameters,
            retain_graph=index < 2,
            allow_unused=True,
        )
    norms = {
        name: float(torch.sqrt(_dot(values, values)).detach().cpu())
        for name, values in gradients.items()
    }
    cosines: dict[str, float] = {}
    for left, right in (("event", "time"), ("event", "position"), ("time", "position")):
        denominator = max(norms[left] * norms[right], torch.finfo(torch.float32).eps)
        value = float((_dot(gradients[left], gradients[right]) / denominator).detach().cpu())
        if not math.isfinite(value):
            raise RuntimeError("Non-finite gradient cosine")
        cosines[f"{left}_{right}"] = max(-1.0, min(1.0, value))
    result = {
        "sample_ids": list(batch["sample_ids"]),
        "raw_loss": {name: float(value.detach().cpu()) for name, value in components.items()},
        "weights": {name: float(joint_loss_weights[name]) for name in components},
        "raw_gradient_norm": norms,
        "effective_gradient_norm": {
            name: norms[name] * float(joint_loss_weights[name])
            for name in norms
        },
        "gradient_cosine": cosines,
    }
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return result


def diagnose_checkpoint(checkpoint: Path, device_name: str) -> dict[str, Any]:
    """Reproduce initialization and best diagnostics for an existing joint run."""

    # Imported lazily because training imports the core diagnostic function above.
    from .training import (  # pylint: disable=import-outside-toplevel
        TargetTrainingConfig,
        _loader,
        backbone_state_hash,
        set_seed,
    )

    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    raw_config = dict(state["config"])
    accepted = {field.name for field in fields(TargetTrainingConfig)}
    raw_config = {key: value for key, value in raw_config.items() if key in accepted}
    for key in ("artifact_path", "output_dir", "sample_plan_path"):
        if raw_config.get(key) is not None:
            raw_config[key] = Path(raw_config[key])
    raw_config["device"] = device_name
    weights = raw_config.get("joint_loss_weights") or {
        "event": 1.0,
        "time": 1.0,
        "position": 1.0,
    }
    raw_config["joint_loss_weights"] = weights
    config = TargetTrainingConfig(**raw_config)
    if config.task != "joint" or config.joint_methods is None:
        raise ValueError("Gradient diagnostics require a joint checkpoint")

    artifacts = ProtocolArtifacts.load(config.artifact_path)
    set_seed(config.seed)
    model = build_target_model(
        artifacts,
        config.task,
        config.method,
        config.joint_methods,
        graph_variant=config.graph_variant,
        possession_topology=config.possession_topology,
        possession_feature_level=config.possession_feature_level,
    ).to(device)
    reproduced_hash = backbone_state_hash(model)
    expected_hash = state.get("initial_backbone_sha256")
    if expected_hash is not None and reproduced_hash != expected_hash:
        raise RuntimeError("Could not reproduce checkpoint backbone initialization")
    loader = _loader("validation", config, artifacts, False)
    batch = move_batch_to_device(next(iter(loader)), device)
    initial = compute_gradient_diagnostics(
        model,
        batch,
        artifacts,
        config.method,
        config.joint_methods,
        weights,
    )
    model.load_state_dict(state["model"])
    best = compute_gradient_diagnostics(
        model,
        batch,
        artifacts,
        config.method,
        config.joint_methods,
        weights,
    )
    return {
        "checkpoint": str(checkpoint),
        "initial_backbone_sha256": reproduced_hash,
        "weights": weights,
        "initial": initial,
        "best": best,
    }
