"""Selection, diagnostics, and paired reporting for joint CE loss scaling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import CONFIRMATION_SEEDS, EXPERIMENT_ROOT, LOSS_BALANCE_SCALES
from .diagnostics import diagnose_checkpoint
from .reporting import _match_statistics, _paired_hierarchical_bootstrap


def scale_label(scale: float) -> str:
    return f"j2_{int(round(scale * 100)):03d}"


def _read_result(path: Path) -> dict[str, Any]:
    return json.loads((path / "result.json").read_text(encoding="utf-8"))


def _metrics(result: dict[str, Any], split: str = "validation") -> dict[str, float]:
    values = result[split]
    return {
        "event_accuracy": float(values["event"]["accuracy"]),
        "event_macro_f1": float(values["event"]["macro_f1"]),
        "time_mae_seconds": float(values["time"]["mae_seconds"]),
        "position_distance_mae_m": float(values["position"]["distance_mae_m"]),
    }


def _variant_dir(variant: str, seed: int) -> Path:
    if variant == "j0":
        return EXPERIMENT_ROOT / "final/joint_original" / f"seed{seed}"
    if variant == "j1":
        return EXPERIMENT_ROOT / "final/joint_optimized" / f"seed{seed}"
    return EXPERIMENT_ROOT / "loss_balance_validation" / variant / f"seed{seed}"


def select_loss_balance() -> dict[str, Any]:
    """Select from validation only and persist an auditable gate decision."""

    output = EXPERIMENT_ROOT / "loss_balance"
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    hashes: dict[int, str] = {}
    variants = ("j0", "j1", *(scale_label(scale) for scale in LOSS_BALANCE_SCALES))
    for variant in variants:
        for seed in CONFIRMATION_SEEDS:
            result = _read_result(_variant_dir(variant, seed))
            if variant.startswith("j2") and result.get("test_accessed"):
                raise RuntimeError("Loss-balance validation unexpectedly accessed test")
            backbone_hash = result["initial_backbone_sha256"]
            if seed in hashes and hashes[seed] != backbone_hash:
                raise RuntimeError(f"Shared HGT initialization differs for seed {seed}")
            hashes[seed] = backbone_hash
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    **_metrics(result),
                    "event_loss_weight": (
                        result["config"].get("joint_loss_weights") or {"event": 1.0}
                    )["event"],
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "validation_by_seed.csv", index=False)
    means = frame.groupby("variant").mean(numeric_only=True)
    means.to_csv(output / "validation_means.csv")
    reference = means.loc["j0"]
    decisions: dict[str, Any] = {}
    eligible: list[str] = []
    for scale in LOSS_BALANCE_SCALES:
        label = scale_label(scale)
        candidate = means.loc[label]
        criteria = {
            "event_macro_f1_improved": bool(
                candidate.event_macro_f1 > reference.event_macro_f1
            ),
            "event_accuracy_within_1pp": bool(
                candidate.event_accuracy >= reference.event_accuracy - 0.01
            ),
            "time_mae_within_0.05s": bool(
                candidate.time_mae_seconds <= reference.time_mae_seconds + 0.05
            ),
            "position_within_0.5m": bool(
                candidate.position_distance_mae_m
                <= reference.position_distance_mae_m + 0.5
            ),
        }
        criteria["eligible"] = all(criteria.values())
        decisions[label] = {
            "event_loss_weight": scale,
            "criteria": criteria,
            "validation_difference_minus_j0": {
                name: float(candidate[name] - reference[name])
                for name in (
                    "event_accuracy",
                    "event_macro_f1",
                    "time_mae_seconds",
                    "position_distance_mae_m",
                )
            },
        }
        if criteria["eligible"]:
            eligible.append(label)
    selected = (
        max(eligible, key=lambda label: float(means.loc[label].event_macro_f1))
        if eligible
        else None
    )
    state = {
        "selection_split": "validation",
        "selected": selected,
        "selected_event_loss_weight": (
            decisions[selected]["event_loss_weight"] if selected else None
        ),
        "test_allowed": selected is not None,
        "decisions": decisions,
        "initial_backbone_sha256_by_seed": hashes,
    }
    (output / "selection.json").write_text(
        json.dumps(state, indent=2), encoding="utf-8"
    )
    return state


def build_reference_diagnostics(device_name: str = "cuda:0") -> Path:
    output = EXPERIMENT_ROOT / "loss_balance/reference_diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    for variant in ("j0", "j1"):
        for seed in CONFIRMATION_SEEDS:
            target = output / f"{variant}_seed{seed}.json"
            if target.exists():
                continue
            checkpoint = _variant_dir(variant, seed) / "best.pt"
            result = diagnose_checkpoint(checkpoint, device_name)
            target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return output


def _diagnostic_rows() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    reference = EXPERIMENT_ROOT / "loss_balance/reference_diagnostics"
    variants = ("j0", "j1", *(scale_label(scale) for scale in LOSS_BALANCE_SCALES))
    for variant in variants:
        for seed in CONFIRMATION_SEEDS:
            if variant in ("j0", "j1"):
                payload = json.loads(
                    (reference / f"{variant}_seed{seed}.json").read_text(encoding="utf-8")
                )
            else:
                payload = _read_result(_variant_dir(variant, seed))["gradient_diagnostics"]
            for point in ("initial", "best"):
                item = payload[point]
                row: dict[str, Any] = {"variant": variant, "seed": seed, "point": point}
                for task, value in item["raw_loss"].items():
                    row[f"raw_loss_{task}"] = value
                for task, value in item["raw_gradient_norm"].items():
                    row[f"raw_grad_{task}"] = value
                for task, value in item["effective_gradient_norm"].items():
                    row[f"effective_grad_{task}"] = value
                for pair, value in item["gradient_cosine"].items():
                    row[f"cosine_{pair}"] = value
                rows.append(row)
    return pd.DataFrame(rows)


def build_loss_balance_report(selection: dict[str, Any] | None = None) -> Path:
    output = EXPERIMENT_ROOT / "loss_balance/report"
    output.mkdir(parents=True, exist_ok=True)
    state = selection or json.loads(
        (EXPERIMENT_ROOT / "loss_balance/selection.json").read_text(encoding="utf-8")
    )
    diagnostics = _diagnostic_rows()
    diagnostics.to_csv(output / "gradient_diagnostics.csv", index=False)
    diagnostics.groupby(["variant", "point"]).mean(numeric_only=True).to_csv(
        output / "gradient_diagnostics_means.csv"
    )
    report: dict[str, Any] = {
        "selected": state["selected"],
        "test_accessed": False,
        "paired_bootstrap": None,
    }
    if state["selected"] is not None:
        selected = state["selected"]
        statistics: dict[str, list[np.ndarray]] = {"j0": [], "j1": [], selected: []}
        expected_ids: list[str] | None = None
        match_ids: list[int] | None = None
        test_rows: list[dict[str, Any]] = []
        for seed in CONFIRMATION_SEEDS:
            for variant in statistics:
                path = (
                    _variant_dir(variant, seed)
                    if variant in ("j0", "j1")
                    else EXPERIMENT_ROOT / "loss_balance_test" / variant / f"seed{seed}"
                )
                result = _read_result(path)
                if not result.get("test_accessed") or result.get("test") is None:
                    raise RuntimeError(f"Missing locked test evaluation: {path}")
                test_rows.append({"variant": variant, "seed": seed, **_metrics(result, "test")})
                frame = pd.read_parquet(path / "test_predictions.parquet").sort_values("sample_id")
                ids = frame.sample_id.tolist()
                if expected_ids is None:
                    expected_ids = ids
                    match_ids = sorted(frame.match_id.astype(int).unique().tolist())
                elif ids != expected_ids:
                    raise RuntimeError("Test prediction sample IDs are not paired")
                statistics[variant].append(_match_statistics(frame, match_ids or []))
        pd.DataFrame(test_rows).to_csv(output / "test_by_seed.csv", index=False)
        bootstrap = {
            f"{selected}_minus_{reference}": _paired_hierarchical_bootstrap(
                np.stack(statistics[reference]), np.stack(statistics[selected])
            )
            for reference in ("j0", "j1")
        }
        (output / "paired_bootstrap_test.json").write_text(
            json.dumps({"iterations": 10_000, "comparisons": bootstrap}, indent=2),
            encoding="utf-8",
        )
        report.update(test_accessed=True, paired_bootstrap=bootstrap)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output
