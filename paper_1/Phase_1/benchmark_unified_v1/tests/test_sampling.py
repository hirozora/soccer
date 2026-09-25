from __future__ import annotations

from dataclasses import dataclass

from football_benchmark.data import CanonicalEventDataset, load_records
from football_benchmark.sampling import (
    FEASIBILITY_SAMPLE_SEED,
    TargetSamplePlan,
    build_feasibility_sample_plan,
    stratified_transition_indices,
)
from football_benchmark.constants import FEASIBILITY_SAMPLE_PLAN_PATH


@dataclass(frozen=True)
class RecordStub:
    match_id: int
    num_events: int


def test_stratified_sampling_is_deterministic_and_covers_each_bin() -> None:
    selected = stratified_transition_indices(
        1700,
        128,
        seed=FEASIBILITY_SAMPLE_SEED,
        split="train",
        match_id=123,
    )
    repeated = stratified_transition_indices(
        1700,
        128,
        seed=FEASIBILITY_SAMPLE_SEED,
        split="train",
        match_id=123,
    )
    assert selected == repeated
    assert len(selected) == len(set(selected)) == 128
    for stratum, current in enumerate(selected):
        assert (stratum * 1700) // 128 <= current
        assert current < ((stratum + 1) * 1700) // 128


def test_feasibility_plan_samples_train_validation_and_keeps_test() -> None:
    records = {
        "train": [RecordStub(1, 301), RecordStub(2, 201)],
        "validation": [RecordStub(3, 151)],
        "test": [RecordStub(4, 141)],
    }
    plan = build_feasibility_sample_plan(records, targets_per_match=128)
    assert plan.sample_count("train") == 256
    assert plan.sample_count("validation") == 128
    assert plan.sample_count("test") == 140
    assert plan.currents_by_match("test")[4] == tuple(range(140))


def test_sample_plan_round_trip(tmp_path) -> None:
    records = {
        "train": [RecordStub(1, 20)],
        "validation": [RecordStub(2, 20)],
        "test": [RecordStub(3, 20)],
    }
    original = build_feasibility_sample_plan(records, targets_per_match=8)
    restored = TargetSamplePlan.load(original.save(tmp_path / "plan.json"))
    assert restored == original


def test_real_feasibility_plan_has_fixed_split_sizes() -> None:
    plan = build_feasibility_sample_plan(
        {split: load_records(split) for split in ("train", "validation", "test")}
    )
    assert plan.sample_count("train") == 266 * 128
    assert plan.sample_count("validation") == 57 * 128
    assert plan.sample_count("test") == sum(
        record.num_events - 1 for record in load_records("test")
    )


def test_dataset_uses_selected_targets_without_changing_history(artifacts) -> None:
    record = load_records("train")[:1]
    selected = {record[0].match_id: (0, 100, 500)}
    dataset = CanonicalEventDataset(
        record,
        artifacts,
        window_size=80,
        selected_currents=selected,
    )
    assert len(dataset) == 3
    assert [dataset[index].current_event_index for index in range(3)] == [0, 100, 500]
    sample = dataset[2]
    assert sample.stop == 501
    assert sample.start == 421
    assert sample.target_event_index == 501


def test_built_feasibility_artifacts_match_fixed_plan(feasibility_artifacts) -> None:
    plan = TargetSamplePlan.load(FEASIBILITY_SAMPLE_PLAN_PATH)
    assert feasibility_artifacts.metadata["samples"] == plan.sample_count("train")
    assert feasibility_artifacts.metadata["samples"] == 34_048
    assert plan.sample_count("validation") == 7_296
    assert plan.sample_count("test") == 96_854
    active = feasibility_artifacts.fine_active_mask
    sums = feasibility_artifacts.fold_matrix[:, active].sum(dim=0)
    assert bool((sums - 1).abs().max() < 1e-7)
