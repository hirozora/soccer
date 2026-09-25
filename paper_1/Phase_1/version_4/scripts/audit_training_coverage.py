"""Read-only audit of fixed supervision targets versus F80 input coverage."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import statistics

import torch


V4 = Path(__file__).resolve().parents[1]
PHASE = V4.parent
PLAN = PHASE / "benchmark_unified_v1/artifacts/feasibility/sample_plan.json"
SPLITS = PHASE / "version_1/data_splits/temporal_match_split_v1.csv"
GRAPHS = PHASE / "data/whyscout/processed/heterogeneous_graphs/semantic_v3_possession"
NAMES = (
    "Duel", "Foul", "Free Kick", "Goalkeeper leaving line", "Interruption",
    "Offside", "Others on the ball", "Pass", "Save attempt", "Shot",
)


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def history_coverage(n, anchors, window=80):
    assert anchors == sorted(set(anchors))
    assert all(0 <= a < n - 1 for a in anchors)
    covered = torch.zeros(n, dtype=torch.bool)
    counts = []
    for anchor in anchors:
        start = max(0, anchor - window + 1)
        covered[start:anchor + 1] = True
        counts.append(anchor - start + 1)
    return covered, counts


def self_test():
    covered, sizes = history_coverage(7, [0, 4], 3)
    assert covered.tolist() == [True, False, True, True, True, False, False]
    assert sizes == [1, 3]
    covered, _ = history_coverage(4, [0, 1, 2], 80)
    assert covered.tolist() == [True, True, True, False]
    try:
        history_coverage(4, [3])
    except AssertionError:
        pass
    else:
        raise AssertionError("Target event must not be an anchor")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pct(a, b):
    return 100 * a / b if b else 0.0


def audit(output):
    torch.set_num_threads(1)
    self_test()
    plan = json.loads(PLAN.read_text())
    records = [r for r in csv.DictReader(SPLITS.open()) if r["competition_slug"] == "England"]
    provenance = {str(PLAN): digest(PLAN), str(SPLITS): digest(SPLITS)}
    configs = []
    for seed in range(20260715, 20260718):
        path = V4 / f"experiments/layerwise_partial_sharing_v1/training/partial_l2/seed{seed}/result.json"
        result = json.loads(path.read_text())
        cfg = result["config"]
        assert Path(cfg["sample_plan_path"]).resolve() == PLAN.resolve()
        assert cfg["max_train_samples"] is None and cfg["max_validation_samples"] is None
        assert result["training_complete"] and result["epochs_completed"] == 24
        configs.append({"seed": seed, "epochs": 24, "sample_plan_sha256": digest(PLAN)})
        provenance[str(path)] = digest(path)

    summaries, classes, matches, players, periods = [], [], [], [], []
    for split in ("train", "validation"):
        full_types, chosen_types, input_types = Counter(), Counter(), Counter()
        full_players, chosen_players, input_players = Counter(), Counter(), Counter()
        full_periods, chosen_periods = Counter(), Counter()
        window_sizes = []
        selected_records = [r for r in records if r["split"] == split]
        selected_map = plan["selections"][split]
        assert set(selected_map) == {r["match_id"] for r in selected_records}
        total_events = input_events = total_targets = chosen_targets = 0
        full_time = chosen_time = full_position = chosen_position = 0
        cross_period_windows = 0
        for index, record in enumerate(selected_records):
            path = GRAPHS / record["graph_path"]
            provenance[str(path)] = digest(path)
            graph = torch.load(path, map_location="cpu", weights_only=False)
            event = graph["node_stores"]["event"]
            n = int(event["num_nodes"])
            assert n == int(record["num_events"])
            anchors = selected_map[record["match_id"]]
            assert len(anchors) == min(n - 1, plan["targets_per_match"])
            chosen = torch.tensor(anchors, dtype=torch.long) + 1
            covered, sizes = history_coverage(n, anchors)
            window_sizes.extend(sizes)
            types = event["event_type_index"]
            ids = graph["node_stores"]["player"]["raw_id"][event["player_local_index"]]
            assert int(types.min()) >= 0 and int(types.max()) < 10
            period = event["period_index"]
            for a in anchors:
                cross_period_windows += int(period[a] != period[max(0, a - 79)])
            full_types.update(types[1:].tolist())
            chosen_types.update(types[chosen].tolist())
            input_types.update(types[covered].tolist())
            full_players.update(p for p in ids[1:].tolist() if p > 0)
            chosen_players.update(p for p in ids[chosen].tolist() if p > 0)
            input_players.update(p for p in ids[covered].tolist() if p > 0)
            full_periods.update(period[1:].tolist())
            chosen_periods.update(period[chosen].tolist())
            valid_time = period[1:] == period[:-1]
            full_time += int(valid_time.sum())
            chosen_time += int(valid_time[chosen - 1].sum())
            position = event["start_position_mask"].bool()
            full_position += int(position[1:].sum())
            chosen_position += int(position[chosen].sum())
            total_events += n
            input_events += int(covered.sum())
            total_targets += n - 1
            chosen_targets += len(chosen)
            matches.append({"split": split, "match_id": record["match_id"],
                            "full_targets": n - 1, "selected_targets": len(chosen),
                            "target_coverage_pct": pct(len(chosen), n - 1),
                            "input_event_union": int(covered.sum()),
                            "possible_input_events": n - 1,
                            "input_coverage_pct": pct(int(covered.sum()), n - 1)})
            if (index + 1) % 50 == 0:
                print(f"{split}: {index + 1}/{len(selected_records)} matches", flush=True)

        expected = {"train": (449025, 34048), "validation": (96891, 7296)}[split]
        assert (total_targets, chosen_targets) == expected
        for label, name in enumerate(NAMES):
            classes.append({"split": split, "event": name,
                            "full_targets": full_types[label], "selected_targets": chosen_types[label],
                            "coverage_pct": pct(chosen_types[label], full_types[label]),
                            "full_share_pct": pct(full_types[label], total_targets),
                            "selected_share_pct": pct(chosen_types[label], chosen_targets),
                            "input_event_union": input_types[label]})
        for p in sorted(full_players):
            players.append({"split": split, "player_id": p, "full_targets": full_players[p],
                            "selected_targets": chosen_players[p], "input_events": input_players[p]})
        for period in sorted(full_periods):
            periods.append({"split": split, "period_index": period,
                            "full_targets": full_periods[period], "selected_targets": chosen_periods[period]})
        counts = [chosen_players[p] for p in full_players]
        summary = {
            "split": split, "matches": len(selected_records), "full_targets": total_targets,
            "selected_targets": chosen_targets, "target_coverage_pct": pct(chosen_targets, total_targets),
            "raw_events": total_events, "input_event_union": input_events,
            "possible_input_events": total_targets, "input_coverage_pct": pct(input_events, total_targets),
            "mean_window_events": statistics.mean(window_sizes), "cross_period_windows": cross_period_windows,
            "full_time_valid": full_time, "selected_time_valid": chosen_time,
            "full_position_valid": full_position, "selected_position_valid": chosen_position,
            "full_target_players_raw_id_positive": len(full_players),
            "selected_target_players_raw_id_positive": len(chosen_players),
            "target_players_without_supervision": sum(c == 0 for c in counts),
            "target_players_under_5_supervised": sum(c < 5 for c in counts),
            "target_players_under_10_supervised": sum(c < 10 for c in counts),
            "median_supervised_targets_per_player": statistics.median(counts),
            "input_players_raw_id_positive": len(input_players),
        }
        summaries.append(summary)
        print(json.dumps(summary), flush=True)

    # Test coverage uses the existing split manifest and sampling indices only.
    # No test graph, label, checkpoint prediction, or loader is opened here.
    test_records = [r for r in records if r["split"] == "test"]
    for r in test_records:
        assert plan["selections"]["test"][r["match_id"]] == list(range(int(r["num_events"]) - 1))
    test_meta = {"matches": len(test_records),
                 "targets": sum(int(r["num_events"]) - 1 for r in test_records),
                 "target_coverage_pct": 100, "source": "split manifest and sample plan only"}
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in (("summary", summaries), ("event_coverage", classes),
                       ("match_coverage", matches), ("player_coverage", players), ("period_coverage", periods)):
        write_csv(output / f"{name}.csv", rows)
    report = {"summary": summaries, "test_metadata_only": test_meta, "baseline_configs": configs,
              "sampling_seed": plan["sampling_seed"], "targets_per_match": plan["targets_per_match"],
              "window": 80, "input_hashes": provenance, "script_sha256": digest(Path(__file__)),
              "notes": ["Coverage of supervised targets is not coverage of history inputs.",
                        "Raw player ID > 0 is not the candidate-valid player loss mask.",
                        "All epochs and model seeds reuse the same selected targets.",
                        "Only train/validation graphs are read. Test coverage is metadata-only."]}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Fixed Training Coverage Audit", "", "No training launched. Existing artifacts are read-only.", "",
             "| Split | Matches | Full targets | Selected | Target coverage | F80 input coverage |",
             "|---|---:|---:|---:|---:|---:|"]
    for s in summaries:
        lines.append(f"| {s['split']} | {s['matches']} | {s['full_targets']} | {s['selected_targets']} | "
                     f"{s['target_coverage_pct']:.3f}% | {s['input_coverage_pct']:.3f}% |")
    lines += ["", "Input coverage counts unique events appearing in at least one F80 history,",
              "divided by events that can precede a target (all except the final event per match).",
              "This does not imply that every possible anchor/context/target combination is trained.", "",
              "## Event Supervision", "", "| Split | Event | Full targets | Selected | Coverage |",
              "|---|---|---:|---:|---:|"]
    for c in classes:
        lines.append(f"| {c['split']} | {c['event']} | {c['full_targets']} | {c['selected_targets']} | {c['coverage_pct']:.2f}% |")
    lines += ["", "Player coverage uses positive raw actor IDs, not the candidate-valid loss mask.",
              "See player_coverage.csv for zero/low-supervision players and report.json for provenance.",
              "Three seeds and 24 epochs repeat the same target subset; they do not expand unique target coverage."]
    (output / "README.md").write_text("\n".join(lines) + "\n")
    print(f"Report: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=V4 / "experiments/training_coverage_audit_v1")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("Coverage unit checks passed")
    else:
        audit(args.output)
