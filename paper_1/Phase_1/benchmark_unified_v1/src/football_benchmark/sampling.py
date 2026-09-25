"""Deterministic target sampling for the lightweight feasibility benchmark."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch


FEASIBILITY_TARGETS_PER_MATCH = 128
FEASIBILITY_SAMPLE_SEED = 20260701


def _match_seed(seed: int, split: str, match_id: int) -> int:
    payload = f"{seed}:{split}:{match_id}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def stratified_transition_indices(
    num_transitions: int,
    budget: int,
    *,
    seed: int,
    split: str,
    match_id: int,
) -> tuple[int, ...]:
    """Draw one target transition from each contiguous temporal stratum."""

    if num_transitions < 1:
        return ()
    if budget < 1:
        raise ValueError("budget must be positive")
    count = min(num_transitions, budget)
    generator = torch.Generator().manual_seed(_match_seed(seed, split, match_id))
    selected: list[int] = []
    for stratum in range(count):
        start = (stratum * num_transitions) // count
        stop = ((stratum + 1) * num_transitions) // count
        offset = int(torch.randint(stop - start, (1,), generator=generator))
        selected.append(start + offset)
    return tuple(selected)


@dataclass(frozen=True)
class TargetSamplePlan:
    """Immutable current-event indices shared by all model families."""

    competition: str
    sampling_seed: int
    targets_per_match: int
    selections: dict[str, dict[int, tuple[int, ...]]]
    split_modes: dict[str, str]

    def currents_by_match(self, split: str) -> Mapping[int, tuple[int, ...]]:
        if split not in self.selections:
            raise KeyError(f"Split is absent from sample plan: {split}")
        return self.selections[split]

    def sample_count(self, split: str) -> int:
        return sum(len(values) for values in self.currents_by_match(split).values())

    def state_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "competition": self.competition,
            "sampling_seed": self.sampling_seed,
            "targets_per_match": self.targets_per_match,
            "split_modes": self.split_modes,
            "selections": {
                split: {str(match_id): list(values) for match_id, values in matches.items()}
                for split, matches in self.selections.items()
            },
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "TargetSamplePlan":
        raw_selections = state["selections"]
        if not isinstance(raw_selections, dict):
            raise TypeError("sample plan selections must be a dictionary")
        selections = {
            str(split): {
                int(match_id): tuple(int(index) for index in indices)
                for match_id, indices in matches.items()
            }
            for split, matches in raw_selections.items()
        }
        return cls(
            competition=str(state["competition"]),
            sampling_seed=int(state["sampling_seed"]),
            targets_per_match=int(state["targets_per_match"]),
            selections=selections,
            split_modes={
                str(split): str(mode)
                for split, mode in dict(state["split_modes"]).items()
            },
        )

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.state_dict(), indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "TargetSamplePlan":
        return cls.from_state_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def build_feasibility_sample_plan(
    records_by_split: Mapping[str, Sequence[object]],
    *,
    targets_per_match: int = FEASIBILITY_TARGETS_PER_MATCH,
    sampling_seed: int = FEASIBILITY_SAMPLE_SEED,
    competition: str = "England",
) -> TargetSamplePlan:
    """Sample train/validation targets and retain every test transition."""

    selections: dict[str, dict[int, tuple[int, ...]]] = {}
    split_modes: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        records = records_by_split[split]
        matches: dict[int, tuple[int, ...]] = {}
        for record in records:
            match_id = int(getattr(record, "match_id"))
            num_transitions = int(getattr(record, "num_events")) - 1
            if split == "test":
                selected = tuple(range(num_transitions))
            else:
                selected = stratified_transition_indices(
                    num_transitions,
                    targets_per_match,
                    seed=sampling_seed,
                    split=split,
                    match_id=match_id,
                )
            matches[match_id] = selected
        selections[split] = matches
        split_modes[split] = "all" if split == "test" else "temporal_stratified"
    return TargetSamplePlan(
        competition=competition,
        sampling_seed=sampling_seed,
        targets_per_match=targets_per_match,
        selections=selections,
        split_modes=split_modes,
    )
