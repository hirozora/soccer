"""Fixed-budget training with dual checkpoint selection and exact continuation."""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_

from football_benchmark.protocol import ProtocolArtifacts

from .actor_training import compute_actor_metrics
from .age_pooling_diagnostics import (
    age_pooling_snapshot,
    collect_age_pooling_diagnostics,
)
from .age_propagation_diagnostics import (
    collect_propagation_diagnostics,
    propagation_snapshot,
)
from .constants import FEASIBILITY_ARTIFACT, SAMPLE_PLAN
from .five_task_diagnostics import FIVE_TASK_HEAD_PREFIXES
from .five_task_study import FIVE_TASK_NAMES, FIVE_TASK_WEIGHTS
from .five_task_training import (
    FiveTaskTrainingConfig,
    _fusion_record,
    _loader,
    _merge_prediction_frames,
)
from .fixed_budget_loss import fixed_budget_loss
from .fixed_budget_study import ALL_TASKS, CONFIGURATIONS
from .metrics import compute_metrics
from .model import (
    AgePropagationPartialL2HGT,
    AgePoolingPartialL2HGT,
    FiveTaskViewHGT,
    build_age_propagation_model,
    build_age_pooling_model,
    build_five_task_model,
    build_partial_l2_model,
    build_receptive_field_model,
    build_state_aware_partial_l2_model,
)
from .state_gating_diagnostics import collect_gate_diagnostics, gate_snapshot
from .training import _move_batch_to_device, _write_metric_tables, set_seed


@dataclass(frozen=True)
class FixedBudgetConfig:
    configuration: str
    output_dir: Path
    seed: int
    device: str
    training_budget: int = 16
    learning_rate: float = 9e-4
    batch_size: int = 256
    num_workers: int = 2
    artifact_path: Path = FEASIBILITY_ARTIFACT
    sample_plan_path: Path | None = SAMPLE_PLAN
    max_train_samples: int | None = None
    max_validation_samples: int | None = None
    max_test_samples: int | None = None
    evaluate_test: bool = False
    full_test: bool = False
    weight_decay: float = 1e-4
    gradient_clip: float = 5.0
    resume_from: Path | None = None
    secondary_guard_reference_path: Path | None = None

    def validate(self) -> None:
        if self.configuration not in CONFIGURATIONS:
            raise ValueError(f"Unknown configuration {self.configuration!r}")
        if self.training_budget not in {1, 2, 16, 24}:
            raise ValueError("Training budget must be smoke (1/2), 16, or 24")
        if self.full_test and not self.evaluate_test:
            raise ValueError("full_test requires evaluate_test")

    @property
    def definition(self) -> dict[str, Any]:
        return CONFIGURATIONS[self.configuration]

    @property
    def mode(self) -> str:
        return str(self.definition["mode"])

    @property
    def active_tasks(self) -> tuple[str, ...]:
        return tuple(self.definition["active_tasks"])

    @property
    def checkpoint_metric(self) -> str:
        return str(self.definition["checkpoint_metric"])

    @property
    def player_adapter(self) -> bool:
        return bool(self.definition.get("player_adapter", False))

    @property
    def partial_l2(self) -> bool:
        return bool(self.definition.get("partial_l2", False))

    @property
    def state_gating(self) -> bool:
        return bool(self.definition.get("state_gating", False))

    @property
    def receptive_field(self) -> bool:
        return bool(self.definition.get("receptive_field", False))

    @property
    def age_pooling(self) -> str | None:
        value = self.definition.get("age_pooling")
        return None if value is None else str(value)

    @property
    def age_propagation(self) -> str | None:
        value = self.definition.get("age_propagation")
        return None if value is None else str(value)


def _source_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _common_hash(model: FiveTaskViewHGT) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith(
            (
                "fusion_logits.",
                "player_adapter.",
                "player_convolution.",
                "player_norms.",
                "player_context_projection.",
                "convolutions.0.gate_controller.",
                "convolutions.1.gate_controller.",
                "age_embedding.",
                "pooling_embeddings",
                "age_scorer.",
                "propagation_residuals.",
                "propagation_gate.",
            )
        ):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _rng_state(train_loader: Any) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "sampler": train_loader.sampler.generator.get_state(),
        "loader": train_loader.generator.get_state(),
    }


def _restore_rng(state: dict[str, Any], train_loader: Any) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    train_loader.sampler.generator.set_state(state["sampler"].cpu())
    train_loader.generator.set_state(state["loader"].cpu())


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _loader_config(config: FixedBudgetConfig) -> FiveTaskTrainingConfig:
    return FiveTaskTrainingConfig(
        mode=config.mode,
        output_dir=config.output_dir,
        seed=config.seed,
        device=config.device,
        max_epochs=config.training_budget,
        patience=config.training_budget + 1,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        artifact_path=config.artifact_path,
        sample_plan_path=config.sample_plan_path,
        max_train_samples=config.max_train_samples,
        max_validation_samples=config.max_validation_samples,
        max_test_samples=config.max_test_samples,
        evaluate_test=config.evaluate_test,
        full_test=config.full_test,
        weight_decay=config.weight_decay,
        gradient_clip=config.gradient_clip,
        receptive_field_configuration=(config.configuration if config.receptive_field else None),
    )


def _inactive_prefixes(active_tasks: tuple[str, ...]) -> tuple[str, ...]:
    mapping = {
        "event": "event_head.",
        "time": "time_head.",
        "position": "position_head.",
        "team": "team_actor_head.",
        "player": "player_actor_scorer.",
    }
    return tuple(prefix for task, prefix in mapping.items() if task not in active_tasks)


def _optimizer(model: FiveTaskViewHGT, config: FixedBudgetConfig) -> torch.optim.Optimizer:
    inactive = _inactive_prefixes(config.active_tasks)
    base, no_decay = [], []
    for name, parameter in model.named_parameters():
        if name.startswith(inactive):
            parameter.requires_grad_(False)
            continue
        if name.startswith("player_adapter.") and "player" not in config.active_tasks:
            parameter.requires_grad_(False)
            continue
        (no_decay if name.startswith("fusion_logits.") else base).append(parameter)
    groups: list[dict[str, Any]] = [{"params": base, "weight_decay": config.weight_decay}]
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, lr=config.learning_rate)


def _all_components(
    predictions: dict[str, torch.Tensor], batch: dict[str, Any], artifacts: ProtocolArtifacts
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    return fixed_budget_loss(predictions, batch, artifacts, ALL_TASKS)


@torch.no_grad()
@torch.no_grad()
def evaluate_fixed_budget(
    model: FiveTaskViewHGT,
    loader: Any,
    artifacts: ProtocolArtifacts,
    active_tasks: tuple[str, ...],
    device: torch.device,
) -> tuple[dict[str, Any], pd.DataFrame]:
    model.eval()
    frames: list[pd.DataFrame] = []
    sums = {name: 0.0 for name in FIVE_TASK_NAMES}
    active_sum = core_sum = 0.0
    count = 0
    for raw_batch in loader:
        batch = _move_batch_to_device(raw_batch, device)
        predictions = model(batch)
        _, components, core = _all_components(predictions, batch, artifacts)
        active = sum(components[name] * FIVE_TASK_WEIGHTS[name] for name in active_tasks) / 3.0
        frame, _ = _merge_prediction_frames(predictions, batch, active)
        frames.append(frame)
        size = len(batch["sample_ids"])
        active_sum += float(active) * size
        core_sum += float(core) * size
        for name, value in components.items():
            sums[name] += float(value) * size
        count += size
    frame = pd.concat(frames, ignore_index=True)
    metrics = compute_metrics(frame, active_sum / max(count, 1))
    metrics["team"] = compute_actor_metrics(frame, "team", sums["team"] / max(count, 1))["team"]
    metrics["player"] = compute_actor_metrics(frame, "player", sums["player"] / max(count, 1))["player"]
    metrics["task_losses"] = {name: value / max(count, 1) for name, value in sums.items()}
    metrics["core_etp_loss"] = core_sum / max(count, 1)
    metrics["joint_active_loss"] = active_sum / max(count, 1)
    return metrics, frame


def _train_epoch(
    model: FiveTaskViewHGT,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    artifacts: ProtocolArtifacts,
    config: FixedBudgetConfig,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    sums = {name: 0.0 for name in ("loss", "core", *FIVE_TASK_NAMES)}
    count = 0
    for raw_batch in loader:
        batch = _move_batch_to_device(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss, components, core = fixed_budget_loss(model(batch), batch, artifacts, config.active_tasks)
        loss.backward()
        clip_grad_norm_((p for p in model.parameters() if p.requires_grad), config.gradient_clip)
        optimizer.step()
        size = len(batch["sample_ids"])
        sums["loss"] += float(loss.detach()) * size
        sums["core"] += float(core.detach()) * size
        for name, value in components.items():
            sums[name] += float(value.detach()) * size
        count += size
    return {name: value / max(count, 1) for name, value in sums.items()}


def _gradient_diagnostics(
    model: FiveTaskViewHGT,
    batch: dict[str, Any],
    artifacts: ProtocolArtifacts,
    active_tasks: tuple[str, ...],
) -> dict[str, Any]:
    import itertools

    was_training = model.training
    model.eval()
    _, components, _ = fixed_budget_loss(model(batch), batch, artifacts, active_tasks)
    parameters = [
        value for name, value in model.named_parameters()
        if value.requires_grad
        and not name.startswith(FIVE_TASK_HEAD_PREFIXES)
        and not name.startswith(("fusion_logits.", "player_adapter."))
    ]
    gradients = {}
    for index, task in enumerate(active_tasks):
        gradients[task] = torch.autograd.grad(
            components[task], parameters, retain_graph=index + 1 < len(active_tasks), allow_unused=True
        )

    def dot(left: Any, right: Any) -> torch.Tensor:
        return torch.stack([(a * b).sum() for a, b in zip(left, right) if a is not None and b is not None]).sum()

    norms = {task: float(torch.sqrt(dot(values, values)).detach()) for task, values in gradients.items()}
    cosines = {}
    for left, right in itertools.combinations(active_tasks, 2):
        value = float((dot(gradients[left], gradients[right]) / max(norms[left] * norms[right], 1e-12)).detach())
        cosines[f"{left}_{right}"] = max(-1.0, min(1.0, value))
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return {
        "raw_gradient_norm": norms,
        "effective_gradient_norm": {task: norms[task] * FIVE_TASK_WEIGHTS[task] for task in norms},
        "gradient_cosine": cosines,
    }


def _partial_layer_gradient_diagnostics(
    model: FiveTaskViewHGT,
    batch: dict[str, Any],
    artifacts: ProtocolArtifacts,
    active_tasks: tuple[str, ...],
) -> dict[str, Any]:
    """Report task gradients separately for shared and private HGT regions."""

    if not hasattr(model, "player_convolution"):
        return {}
    import itertools

    was_training = model.training
    model.eval()
    _, components, _ = fixed_budget_loss(model(batch), batch, artifacts, active_tasks)
    named = dict(model.named_parameters())
    head_prefixes = (*FIVE_TASK_HEAD_PREFIXES, "fusion_logits.", "player_adapter.")
    groups = {
        "shared_layer1": [
            value
            for name, value in named.items()
            if name.startswith(("convolutions.0.", "norms.0."))
        ],
        "main_layer2": [
            value
            for name, value in named.items()
            if name.startswith(("convolutions.1.", "norms.1.", "context_projection."))
        ],
        "player_layer2": [
            value
            for name, value in named.items()
            if name.startswith(
                (
                    "player_convolution.",
                    "player_norms.",
                    "player_context_projection.",
                )
            )
        ],
    }
    assigned = {id(value) for values in groups.values() for value in values}
    groups["shared_encoders"] = [
        value
        for name, value in named.items()
        if value.requires_grad
        and id(value) not in assigned
        and not name.startswith(head_prefixes)
    ]

    output: dict[str, Any] = {}
    for group_name, parameters in groups.items():
        gradients = {}
        for index, task in enumerate(active_tasks):
            gradients[task] = torch.autograd.grad(
                components[task],
                parameters,
                retain_graph=True,
                allow_unused=True,
            )

        def dot(left: Any, right: Any) -> torch.Tensor:
            terms = [
                (a * b).sum()
                for a, b in zip(left, right)
                if a is not None and b is not None
            ]
            return torch.stack(terms).sum() if terms else torch.zeros((), device=batch["targets"]["raw_event_10"].device)

        norms = {
            task: float(torch.sqrt(dot(values, values)).detach())
            for task, values in gradients.items()
        }
        cosines = {}
        for left, right in itertools.combinations(active_tasks, 2):
            denominator = norms[left] * norms[right]
            value = 0.0 if denominator <= 1e-12 else float(
                (dot(gradients[left], gradients[right]) / denominator).detach()
            )
            cosines[f"{left}_{right}"] = max(-1.0, min(1.0, value))
        output[group_name] = {
            "raw_gradient_norm": norms,
            "effective_gradient_norm": {
                task: norms[task] * FIVE_TASK_WEIGHTS[task] for task in norms
            },
            "gradient_cosine": cosines,
        }
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return output


def _checkpoint_payload(
    model: FiveTaskViewHGT,
    optimizer: torch.optim.Optimizer,
    config: FixedBudgetConfig,
    epoch: int,
    history: list[dict[str, Any]],
    common_hash: str,
    initial_diagnostics: dict[str, Any],
    initial_layer_diagnostics: dict[str, Any],
    train_loader: Any,
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "history": history,
        "config": asdict(config),
        "mode": config.mode,
        "architecture": (
            "partial_l2_age_propagation"
            if config.age_propagation is not None
            else
            "partial_l2_age_pooling"
            if config.age_pooling is not None
            else "partial_l2_receptive_field"
            if config.receptive_field
            else
            "partial_l2_state_gated"
            if config.state_gating
            else "partial_l2"
            if config.partial_l2
            else "fully_shared"
        ),
        "active_tasks": config.active_tasks,
        "initial_common_sha256": common_hash,
        "initial_gradient_diagnostics": initial_diagnostics,
        "initial_layer_gradient_diagnostics": initial_layer_diagnostics,
        "rng": _rng_state(train_loader),
        "source_sha256": _source_hash(),
    }


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def guarded_core_eligible(
    metrics: dict[str, Any], reference: dict[str, float], *, margin: float = 0.01
) -> bool:
    return bool(
        float(metrics["team"]["accuracy"]) >= reference["team_accuracy"] - margin
        and float(metrics["player"]["top1_accuracy"])
        >= reference["player_top1"] - margin
    )


def _guard_metrics(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("validation_guarded_core") or payload.get("validation")
    if metrics is None:
        raise RuntimeError(f"Guard reference has no validation metrics: {path}")
    return {
        "team_accuracy": float(metrics["team"]["accuracy"]),
        "player_top1": float(metrics["player"]["top1_accuracy"]),
    }


def _evaluate_checkpoint(
    checkpoint: Path,
    model: FiveTaskViewHGT,
    loader: Any,
    artifacts: ProtocolArtifacts,
    config: FixedBudgetConfig,
    device: torch.device,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    metrics, predictions = evaluate_fixed_budget(model, loader, artifacts, config.active_tasks, device)
    return metrics, predictions, state


def run_fixed_budget_training(config: FixedBudgetConfig) -> dict[str, Any]:
    config.validate()
    set_seed(config.seed)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = ProtocolArtifacts.load(config.artifact_path)
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    model = (
        build_age_propagation_model(artifacts, config.age_propagation)
        if config.age_propagation is not None
        else build_age_pooling_model(artifacts, config.age_pooling)
        if config.age_pooling is not None
        else build_receptive_field_model(artifacts, config.configuration)
        if config.receptive_field
        else
        build_state_aware_partial_l2_model(artifacts)
        if config.state_gating
        else build_partial_l2_model(artifacts, config.mode)
        if config.partial_l2
        else build_five_task_model(
            artifacts, config.mode, player_adapter=config.player_adapter
        )
    ).to(device)
    common_hash = _common_hash(model)
    optimizer = _optimizer(model, config)
    loader_config = _loader_config(config)
    train_loader = _loader("train", loader_config, artifacts, True)
    validation_loader = _loader("validation", loader_config, artifacts, False)
    diagnostic_loader = _loader("validation", loader_config, artifacts, False, batch_size=8)
    diagnostic_batch = _move_batch_to_device(next(iter(diagnostic_loader)), device)
    history: list[dict[str, Any]] = []
    start_epoch = 1
    initial_diagnostics = _gradient_diagnostics(model, diagnostic_batch, artifacts, config.active_tasks)
    initial_layer_diagnostics = _partial_layer_gradient_diagnostics(
        model, diagnostic_batch, artifacts, config.active_tasks
    )
    if config.resume_from is not None:
        state = torch.load(config.resume_from, map_location=device, weights_only=False)
        expected_architecture = (
            "partial_l2_age_propagation"
            if config.age_propagation is not None
            else
            "partial_l2_age_pooling"
            if config.age_pooling is not None
            else "partial_l2_receptive_field"
            if config.receptive_field
            else
            "partial_l2_state_gated"
            if config.state_gating
            else "partial_l2"
            if config.partial_l2
            else "fully_shared"
        )
        if state.get("architecture", "fully_shared") != expected_architecture:
            raise ValueError("Resume checkpoint architecture mismatch")
        if state["active_tasks"] != config.active_tasks or state["mode"] != config.mode or int(state["config"]["seed"]) != config.seed:
            raise ValueError("Resume checkpoint identity mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history = list(state["history"])
        start_epoch = int(state["epoch"]) + 1
        initial_diagnostics = state["initial_gradient_diagnostics"]
        initial_layer_diagnostics = state.get("initial_layer_gradient_diagnostics", {})
        _restore_rng(state["rng"], train_loader)
    if start_epoch > config.training_budget:
        raise ValueError("Resume checkpoint already reached requested budget")

    best_core = min((row["core_etp_loss"], row["epoch"]) for row in history) if history else (float("inf"), 0)
    best_joint = min((row["joint_active_loss"], row["epoch"]) for row in history) if history else (float("inf"), 0)
    best_guarded = min(
        (
            (row["core_etp_loss"], row["epoch"])
            for row in history
            if row.get("guarded_core_eligible", False)
        ),
        default=(float("inf"), 0),
    )
    guard_reference = None
    guard_references: list[dict[str, Any]] = []
    if config.checkpoint_metric == "guarded_core":
        from .fixed_budget_study import training_dir

        reference_path = training_dir("five_f80", config.seed) / "result.json"
        if not reference_path.exists():
            raise RuntimeError(f"Missing Five-F80 guard reference: {reference_path}")
        reference = json.loads(reference_path.read_text(encoding="utf-8"))["validation"]
        guard_reference = {
            "team_accuracy": float(reference["team"]["accuracy"]),
            "player_top1": float(reference["player"]["top1_accuracy"]),
        }
        guard_references.append({"name": "five_f80", "margin": 0.01, **guard_reference})
        if config.secondary_guard_reference_path is not None:
            secondary = _guard_metrics(config.secondary_guard_reference_path)
            guard_references.append(
                {"name": "partial_l2_f80", "margin": 0.005, **secondary}
            )
    started = time.monotonic()
    for epoch in range(start_epoch, config.training_budget + 1):
        train = _train_epoch(model, train_loader, optimizer, artifacts, config, device)
        validation, _ = evaluate_fixed_budget(model, validation_loader, artifacts, config.active_tasks, device)
        record: dict[str, Any] = {
            "epoch": epoch,
            "train": train,
            "validation": validation,
            "core_etp_loss": float(validation["core_etp_loss"]),
            "joint_active_loss": float(validation["joint_active_loss"]),
        }
        if guard_reference is not None:
            smoke_guard_bypass = (
                config.training_budget <= 2
                and config.max_train_samples is not None
                and config.max_train_samples <= 128
            )
            record["guarded_core_eligible"] = bool(
                smoke_guard_bypass
                or all(
                    guarded_core_eligible(
                        validation,
                        {
                            "team_accuracy": item["team_accuracy"],
                            "player_top1": item["player_top1"],
                        },
                        margin=float(item["margin"]),
                    )
                    for item in guard_references
                )
            )
            record["guarded_core_smoke_bypass"] = smoke_guard_bypass
        if model.fusion_logits:
            record["fusion"] = _fusion_record(model, diagnostic_batch)
        if isinstance(model, AgePoolingPartialL2HGT):
            record["age_pooling"] = age_pooling_snapshot(model, diagnostic_batch)
        if isinstance(model, AgePropagationPartialL2HGT):
            record["age_propagation"] = propagation_snapshot(
                model, diagnostic_batch
            )
        if config.state_gating:
            record["relation_gates"] = gate_snapshot(model, diagnostic_batch)
        history.append(record)
        payload = _checkpoint_payload(
            model,
            optimizer,
            config,
            epoch,
            history,
            common_hash,
            initial_diagnostics,
            initial_layer_diagnostics,
            train_loader,
        )
        if record["core_etp_loss"] < best_core[0] - 1e-12:
            best_core = (record["core_etp_loss"], epoch)
            _atomic_torch_save(payload, output / "best_core.pt")
        if record["joint_active_loss"] < best_joint[0] - 1e-12:
            best_joint = (record["joint_active_loss"], epoch)
            _atomic_torch_save(payload, output / "best_joint.pt")
        if (
            record.get("guarded_core_eligible", False)
            and record["core_etp_loss"] < best_guarded[0] - 1e-12
        ):
            best_guarded = (record["core_etp_loss"], epoch)
            _atomic_torch_save(payload, output / "best_guarded_core.pt")
        _atomic_torch_save(payload, output / "last.pt")
        _write_json(output / "history.json", history)

    core_metrics, core_predictions, core_state = _evaluate_checkpoint(
        output / "best_core.pt", model, validation_loader, artifacts, config, device
    )
    core_predictions.to_parquet(output / "validation_predictions_core.parquet", index=False)
    joint_metrics, joint_predictions, joint_state = _evaluate_checkpoint(
        output / "best_joint.pt", model, validation_loader, artifacts, config, device
    )
    joint_predictions.to_parquet(output / "validation_predictions_joint.parquet", index=False)
    guarded_metrics = guarded_predictions = guarded_state = None
    if config.checkpoint_metric == "guarded_core":
        guarded_path = output / "best_guarded_core.pt"
        if not guarded_path.exists():
            raise RuntimeError("No epoch satisfied the Team/Player guarded-core constraints")
        guarded_metrics, guarded_predictions, guarded_state = _evaluate_checkpoint(
            guarded_path, model, validation_loader, artifacts, config, device
        )
        guarded_predictions.to_parquet(
            output / "validation_predictions_guarded_core.parquet", index=False
        )
    selected_name = {
        "core_etp": "core",
        "joint_five": "joint",
        "guarded_core": "guarded_core",
    }[config.checkpoint_metric]
    selected_metrics = (
        core_metrics
        if selected_name == "core"
        else joint_metrics
        if selected_name == "joint"
        else guarded_metrics
    )
    selected_predictions = (
        core_predictions
        if selected_name == "core"
        else joint_predictions
        if selected_name == "joint"
        else guarded_predictions
    )
    selected_state = (
        core_state
        if selected_name == "core"
        else joint_state
        if selected_name == "joint"
        else guarded_state
    )
    selected_predictions.to_parquet(output / "validation_predictions.parquet", index=False)
    _write_metric_tables(selected_metrics, output)
    model.load_state_dict(selected_state["model"])
    best_diagnostics = _gradient_diagnostics(model, diagnostic_batch, artifacts, config.active_tasks)
    best_layer_diagnostics = _partial_layer_gradient_diagnostics(
        model, diagnostic_batch, artifacts, config.active_tasks
    )
    gate_diagnostics = None
    if config.state_gating:
        gate_diagnostics = collect_gate_diagnostics(
            model, validation_loader, device
        )
        _write_json(output / "gate_diagnostics.json", gate_diagnostics)
    age_diagnostics = None
    if isinstance(model, AgePoolingPartialL2HGT):
        age_diagnostics = collect_age_pooling_diagnostics(
            model, validation_loader, device
        )
        _write_json(output / "age_pooling_diagnostics.json", age_diagnostics)
    propagation_diagnostics = None
    if isinstance(model, AgePropagationPartialL2HGT):
        propagation_diagnostics = collect_propagation_diagnostics(
            model, diagnostic_batch
        )
        _write_json(
            output / "age_propagation_diagnostics.json",
            propagation_diagnostics,
        )
    result = {
        "config": asdict(config),
        "definition": config.definition,
        "training_complete": len(history) == config.training_budget,
        "epochs_completed": len(history),
        "best_core_epoch": int(core_state["epoch"]),
        "best_joint_epoch": int(joint_state["epoch"]),
        "selected_checkpoint": selected_name,
        "best_epoch": int(selected_state["epoch"]),
        "history": history,
        "validation": selected_metrics,
        "validation_core": core_metrics,
        "validation_joint": joint_metrics,
        "validation_guarded_core": guarded_metrics,
        "guard_reference": guard_reference,
        "guard_references": guard_references,
        "test": None,
        "elapsed_seconds": time.monotonic() - started,
        "parameters": sum(value.numel() for value in model.parameters()),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "initial_common_sha256": common_hash,
        "gradient_diagnostics": {"initial": initial_diagnostics, "best": best_diagnostics},
        "layer_gradient_diagnostics": {
            "initial": initial_layer_diagnostics,
            "best": best_layer_diagnostics,
        },
        "gate_diagnostics": gate_diagnostics,
        "age_pooling_diagnostics": age_diagnostics,
        "age_propagation_diagnostics": propagation_diagnostics,
        "test_accessed": False,
        "source_sha256": _source_hash(),
    }
    _write_json(output / "result.json", result)
    _write_json(output / f"complete_epoch_{config.training_budget}.json", {"complete": True, "epoch": config.training_budget})
    return result


def evaluate_fixed_budget_checkpoint(config: FixedBudgetConfig, checkpoint: Path) -> dict[str, Any]:
    config.validate()
    set_seed(config.seed)
    artifacts = ProtocolArtifacts.load(config.artifact_path)
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    expected_architecture = (
        "partial_l2_age_propagation"
        if config.age_propagation is not None
        else
        "partial_l2_age_pooling"
        if config.age_pooling is not None
        else "partial_l2_receptive_field"
        if config.receptive_field
        else
        "partial_l2_state_gated"
        if config.state_gating
        else "partial_l2"
        if config.partial_l2
        else "fully_shared"
    )
    if state.get("architecture", "fully_shared") != expected_architecture:
        raise ValueError("Evaluation checkpoint architecture mismatch")
    if state["mode"] != config.mode or state["active_tasks"] != config.active_tasks:
        raise ValueError("Evaluation checkpoint identity mismatch")
    model = (
        build_age_propagation_model(artifacts, config.age_propagation)
        if config.age_propagation is not None
        else build_age_pooling_model(artifacts, config.age_pooling)
        if config.age_pooling is not None
        else build_receptive_field_model(artifacts, config.configuration)
        if config.receptive_field
        else
        build_state_aware_partial_l2_model(artifacts)
        if config.state_gating
        else build_partial_l2_model(artifacts, config.mode)
        if config.partial_l2
        else build_five_task_model(
            artifacts, config.mode, player_adapter=config.player_adapter
        )
    ).to(device)
    model.load_state_dict(state["model"])
    split = "test" if config.evaluate_test else "validation"
    metrics, predictions = evaluate_fixed_budget(
        model, _loader(split, _loader_config(config), artifacts, False), artifacts, config.active_tasks, device
    )
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output / f"{split}_predictions.parquet", index=False)
    _write_metric_tables(metrics, output)
    age_diagnostics = None
    if isinstance(model, AgePoolingPartialL2HGT):
        diagnostic_loader = _loader(
            split, _loader_config(config), artifacts, False
        )
        age_diagnostics = collect_age_pooling_diagnostics(
            model, diagnostic_loader, device
        )
        _write_json(output / "age_pooling_diagnostics.json", age_diagnostics)
    propagation_diagnostics = None
    if isinstance(model, AgePropagationPartialL2HGT):
        diagnostic_loader = _loader(
            split, _loader_config(config), artifacts, False, batch_size=8
        )
        diagnostic_batch = _move_batch_to_device(
            next(iter(diagnostic_loader)), device
        )
        propagation_diagnostics = collect_propagation_diagnostics(
            model, diagnostic_batch
        )
        _write_json(
            output / "age_propagation_diagnostics.json",
            propagation_diagnostics,
        )
    result = {
        "config": asdict(config),
        "definition": config.definition,
        "source_checkpoint": str(checkpoint),
        "best_epoch": int(state["epoch"]),
        "validation": metrics if split == "validation" else None,
        "test": metrics if split == "test" else None,
        "test_accessed": split == "test",
        "parameters": sum(value.numel() for value in model.parameters()),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "initial_common_sha256": state["initial_common_sha256"],
        "age_pooling_diagnostics": age_diagnostics,
        "age_propagation_diagnostics": propagation_diagnostics,
        "source_sha256": _source_hash(),
    }
    _write_json(output / "result.json", result)
    return result
