"""Public interfaces for Phase 1 data preparation and graph utilities.

Imports are lazy so graph-independent preprocessing does not require the
PyTorch/PyG training environment to be initialized.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .schema import SCHEMA_VERSION


_LAZY_EXPORTS = {
    "FixedWindowDataset": (".dataset", "FixedWindowDataset"),
    "MatchGraphRecord": (".dataset", "MatchGraphRecord"),
    "load_match_graph": (".dataset", "load_match_graph"),
    "load_match_index": (".dataset", "load_match_index"),
    "sample_fixed_event_window": (".dataset", "sample_fixed_event_window"),
    "slice_event_prefix": (".dataset", "slice_event_prefix"),
    "validate_fixed_event_window": (".dataset", "validate_fixed_event_window"),
    "build_match_graph": (".graph_builder", "build_match_graph"),
    "validate_graph": (".graph_builder", "validate_graph"),
    "build_event_tables": (".event_tables", "build_event_tables"),
    "infer_possessions": (".possession", "infer_possessions"),
    "validate_possessions": (".possession", "validate_possessions"),
    "infer_possessions_v2": (".possession_v2", "infer_possessions_v2"),
    "infer_match_possessions_v2": (
        ".possession_v2",
        "infer_match_possessions_v2",
    ),
    "load_possession_rules_v2": (".possession_v2", "load_possession_rules_v2"),
    "validate_possessions_v2": (".possession_v2", "validate_possessions_v2"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


__all__ = ["SCHEMA_VERSION", *_LAZY_EXPORTS]
