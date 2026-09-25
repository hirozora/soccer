"""Registered fixed-budget multitask conflict experiment."""

from __future__ import annotations

from pathlib import Path

from .constants import EXPERIMENT_ROOT


FIXED_BUDGET_ROOT = EXPERIMENT_ROOT / "five_task_fixed_budget_v1"
CORE_TASKS = ("event", "time", "position")
ALL_TASKS = (*CORE_TASKS, "team", "player")

CONFIGURATIONS = {
    "three_f80": {"mode": "five_f80", "active_tasks": CORE_TASKS, "checkpoint_metric": "core_etp"},
    "three_fixedb": {"mode": "five_hard", "active_tasks": CORE_TASKS, "checkpoint_metric": "core_etp"},
    "three_sfb": {"mode": "five_soft", "active_tasks": CORE_TASKS, "checkpoint_metric": "core_etp"},
    "five_f80": {"mode": "five_f80", "active_tasks": ALL_TASKS, "checkpoint_metric": "joint_five"},
    "five_hard": {"mode": "five_hard", "active_tasks": ALL_TASKS, "checkpoint_metric": "joint_five"},
    "five_soft": {"mode": "five_soft", "active_tasks": ALL_TASKS, "checkpoint_metric": "joint_five"},
    "t4_team": {"mode": "five_f80", "active_tasks": (*CORE_TASKS, "team"), "checkpoint_metric": "core_etp"},
    "t4_player": {"mode": "five_f80", "active_tasks": (*CORE_TASKS, "player"), "checkpoint_metric": "core_etp"},
    "t5_player_adapter": {"mode": "five_f80", "active_tasks": ALL_TASKS, "checkpoint_metric": "core_etp", "player_adapter": True},
    "partial_l2": {
        "mode": "five_f80",
        "active_tasks": ALL_TASKS,
        "checkpoint_metric": "guarded_core",
        "partial_l2": True,
    },
    "partial_l2_hard": {
        "mode": "five_hard",
        "active_tasks": ALL_TASKS,
        "checkpoint_metric": "guarded_core",
        "partial_l2": True,
    },
    "partial_l2_soft": {
        "mode": "five_soft",
        "active_tasks": ALL_TASKS,
        "checkpoint_metric": "guarded_core",
        "partial_l2": True,
    },
    "state_gated_partial_l2": {
        "mode": "five_f80",
        "active_tasks": ALL_TASKS,
        "checkpoint_metric": "guarded_core",
        "partial_l2": True,
        "state_gating": True,
    },
    "rf_core_u5": {
        "mode": "five_f80", "active_tasks": ALL_TASKS,
        "checkpoint_metric": "guarded_core", "partial_l2": True,
        "receptive_field": True,
    },
    "rf_core_u10": {
        "mode": "five_f80", "active_tasks": ALL_TASKS,
        "checkpoint_metric": "guarded_core", "partial_l2": True,
        "receptive_field": True,
    },
    "rf_task": {
        "mode": "five_f80", "active_tasks": ALL_TASKS,
        "checkpoint_metric": "guarded_core", "partial_l2": True,
        "receptive_field": True,
    },
    "ap_shared": {
        "mode": "five_f80", "active_tasks": ALL_TASKS,
        "checkpoint_metric": "guarded_core", "partial_l2": True,
        "age_pooling": "shared",
    },
    "ap_task": {
        "mode": "five_f80", "active_tasks": ALL_TASKS,
        "checkpoint_metric": "guarded_core", "partial_l2": True,
        "age_pooling": "task",
    },
    "pg_constant": {
        "mode": "five_f80", "active_tasks": ALL_TASKS,
        "checkpoint_metric": "guarded_core", "partial_l2": True,
        "age_propagation": "constant",
    },
    "pg_shared": {
        "mode": "five_f80", "active_tasks": ALL_TASKS,
        "checkpoint_metric": "guarded_core", "partial_l2": True,
        "age_propagation": "shared",
    },
    "pg_task": {
        "mode": "five_f80", "active_tasks": ALL_TASKS,
        "checkpoint_metric": "guarded_core", "partial_l2": True,
        "age_propagation": "task",
    },
}

STAGE1 = ("three_f80", "three_fixedb", "three_sfb", "five_f80", "five_hard", "five_soft")
MATCHED_PAIRS = (
    ("three_f80", "five_f80"),
    ("three_fixedb", "five_hard"),
    ("three_sfb", "five_soft"),
)
STAGE2_NEW = ("t4_team", "t4_player")


def training_dir(configuration: str, seed: int) -> Path:
    return FIXED_BUDGET_ROOT / "training" / configuration / f"seed{seed}"


def test_dir(configuration: str, seed: int) -> Path:
    return FIXED_BUDGET_ROOT / "test" / configuration / f"seed{seed}"
