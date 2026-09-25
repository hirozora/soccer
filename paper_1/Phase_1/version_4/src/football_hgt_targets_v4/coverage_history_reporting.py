"""Full-population, match-paired reporting without test-based reselection."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_curve, classification_report, confusion_matrix

from football_benchmark.constants import RAW_EVENT_NAMES
from .coverage_history_training import (ROOT, SEEDS, frame_path, run_dir, require_test,
    evaluate_deployed)
from .coverage_history_probe import probe_dir, evaluate_history_test
from .event_posterior_integration import paired_bootstrap
from .five_task_reporting import _match_statistics, _to_metrics
from .spatiotemporal_reporting import compare, core_loss
from .spatiotemporal_training import write_json


def test_locked(root=ROOT, device="cuda:0"):
    coverage, history = require_test(root)
    if coverage["selected"] == "original" and history["selected"] == "original":
        write_json(root / "selection/test_status.json", {"status": "not_required", "test_accessed": False})
        return
    modes = ["original"]
    if coverage["selected"] != "original":
        modes.append(coverage["selected"])
    if coverage["selected"] == "rotate" and "fixed" in coverage["complete"]:
        modes.append("fixed")
    for mode in modes:
        for seed in SEEDS:
            evaluate_deployed(mode, seed, "test", device, root)
    if history["selected"] != "original":
        for seed in SEEDS:
            evaluate_history_test(seed, device, root)
    write_json(root / "selection/test_status.json", {"status": "completed", "test_accessed": True,
        "coverage_models": modes, "history_selected": history["selected"], "selection_unchanged": True})


def probability_column(frame, index):
    plain = f"event_probability_{index}"
    return plain if plain in frame else f"{plain}_{RAW_EVENT_NAMES[index]}"


def describe_event(frame):
    labels, pred = frame.event_true.to_numpy(int), frame.event_pred.to_numpy(int)
    report = classification_report(labels, pred, labels=list(range(10)), target_names=RAW_EVENT_NAMES,
                                   zero_division=0, output_dict=True)
    target = labels == 5
    if target.any() and (~target).any():
        score = frame[probability_column(frame, 5)].to_numpy()
        fpr, tpr, thresholds = roc_curve(target, score, drop_intermediate=False)
        rare = {"offside_support": int(target.sum()), "offside_pr_auc_average_precision": float(average_precision_score(target, score)),
                "offside_recall_at_fpr_0_001": float(tpr[fpr <= .001].max()),
                "threshold_usage": "descriptive evaluation only; no classification threshold is deployed"}
    else:
        rare = {"offside_support": int(target.sum()), "offside_pr_auc_average_precision": None,
                "offside_recall_at_fpr_0_001": None}
    return {"per_class": report, **rare, "confusion_matrix": confusion_matrix(labels, pred, labels=list(range(10))).tolist()}


def report(root=ROOT):
    coverage = json.loads((root / "selection/coverage_lock.json").read_text())
    history = json.loads((root / "selection/history_lock.json").read_text())
    target = root / "report"; target.mkdir(parents=True, exist_ok=True)
    test_status_path = root / "selection/test_status.json"
    test_status = json.loads(test_status_path.read_text()) if test_status_path.exists() else {"test_accessed": False}
    rows = []
    populations = [("validation", coverage["complete"])]
    if test_status["test_accessed"]:
        require_test(root)
        populations.append(("test", test_status["coverage_models"]))
    for split, modes in populations:
        for mode in modes:
            for seed in SEEDS:
                frame = pd.read_parquet(frame_path(mode, seed, split, root))
                statistics = _to_metrics(_match_statistics(frame, sorted(frame.match_id.unique())).sum(0))
                rows.append({"split": split, "mode": mode, "seed": seed, "core_loss": core_loss(frame),
                             **{k: float(v) for k, v in statistics.items()}})
                write_json(target / split / mode / f"seed{seed}.json", describe_event(frame))
    pd.DataFrame(rows).to_csv(target / "five_task_metrics.csv", index=False)
    numeric = [k for k in rows[0] if k not in ("split", "mode", "seed")]
    pd.DataFrame(rows).groupby(["split", "mode"])[numeric].agg(["mean", "std"]).to_csv(target / "five_task_summary.csv")
    raw_rows = []
    for mode in coverage["complete"]:
        for seed in SEEDS:
            if mode == "original":
                path = root / "references" / f"seed{seed}/original_raw.parquet"
            else:
                import torch
                saved = torch.load(run_dir(mode, seed, root) / "best_guarded_core.pt", map_location="cpu", weights_only=False)
                path = run_dir(mode, seed, root) / f"validation_epoch{saved['epoch']:02d}.parquet"
            frame = pd.read_parquet(path)
            statistics = _to_metrics(_match_statistics(frame, sorted(frame.match_id.unique())).sum(0))
            raw_rows.append({"mode": mode, "seed": seed, **{k: float(v) for k, v in statistics.items()}})
    pd.DataFrame(raw_rows).to_csv(target / "raw_validation_metrics.csv", index=False)
    grouped = []
    for mode in ("null", "team", "team_player", "shuffled_player"):
        for seed in SEEDS:
            frame = pd.read_parquet(probe_dir(mode, seed, root) / "validation.parquet")
            write_json(target / "history_validation" / mode / f"seed{seed}.json", describe_event(frame))
            buckets = pd.cut(frame.team_history_games, [-1, 0, 2, 5, np.inf], labels=["0", "1-2", "3-5", "6+"])
            for name, mask in [(f"team_matches_{v}", buckets == v) for v in buckets.cat.categories] + [
                ("player_history_none", frame.player_history_seen_mass == 0),
                ("player_history_available", frame.player_history_seen_mass > 0)]:
                sub = frame[mask]
                if len(sub):
                    values = describe_event(sub)
                    grouped.append({"mode": mode, "seed": seed, "group": name, "samples": len(sub),
                        "macro_f1": values["per_class"]["macro avg"]["f1-score"],
                        "offside_support": values["offside_support"]})
    pd.DataFrame(grouped).to_csv(target / "history_groups.csv", index=False)
    # Compose selected Event outputs with the unchanged four deployed tasks.
    for split in ("validation", "test"):
        if split == "test" and not test_status["test_accessed"]:
            continue
        for seed in SEEDS:
            frame = pd.read_parquet(frame_path(coverage["selected"], seed, split, root))
            if history["selected"] != "original":
                path = (probe_dir(history["selected"], seed, root) / "validation.parquet" if split == "validation" else
                        root / "history_test" / f"seed{seed}" / f"{history['selected']}.parquet")
                event = pd.read_parquet(path)
                for key in ("sample_id", "match_id", "event_true", "current_event_index"):
                    if not np.array_equal(frame[key], event[key]):
                        raise RuntimeError(f"Composition alignment failure: {key}")
                frame["event_pred"] = event["event_pred"]
                for i in range(10):
                    frame[probability_column(frame, i)] = event[probability_column(event, i)]
                frame["event_confidence"] = event[[probability_column(event, i) for i in range(10)]].max(axis=1)
            output = target / "composed" / split; output.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(output / f"seed{seed}.parquet", index=False)
            stats = _to_metrics(_match_statistics(frame, sorted(frame.match_id.unique())).sum(0))
            write_json(output / f"seed{seed}.json", {k: float(v) for k, v in stats.items()})
    if test_status["test_accessed"]:
        modes = test_status["coverage_models"]
        fs = {m: [pd.read_parquet(frame_path(m, s, "test", root)) for s in SEEDS] for m in modes}
        comparisons = {f"{a}-{b}": compare(fs[b], fs[a]) for a, b in
                       (("fixed", "original"), ("rotate", "original"), ("rotate", "fixed")) if a in fs and b in fs}
        write_json(target / "coverage_test_bootstrap.json", comparisons)
        if history["selected"] != "original":
            modes = ("original", "null", "team") if history["selected"] == "team" else ("original", "null", "team", "team_player", "shuffled_player")
            frames = {m: pd.concat([pd.read_parquet(root / "history_test" / f"seed{s}" / f"{m}.parquet") for s in SEEDS],
                       ignore_index=True).sort_values(["seed", "sample_id"]).reset_index(drop=True) for m in modes}
            pairs = [("team", "null"), ("team", "original"), ("team_player", "null"),
                     ("team_player", "original"), ("team_player", "team"), ("team_player", "shuffled_player")]
            write_json(target / "history_test_bootstrap.json", {f"{a}-{b}": paired_bootstrap(frames[b], frames[a])
                       for a, b in pairs if a in frames and b in frames})
    lines = ["# Coverage and Past-match History", "", f"Validation-locked backbone: **{coverage['selected']}**.",
        f"Validation-locked Event history: **{history['selected']}**.", "",
        "Fixed and Rotate have the same 3192 optimizer steps. Coverage attribution uses Rotate minus Fixed.",
        "History conditions use matches at least 24 hours earlier and never the actual next player/team.",
        "Other tasks retain Position Refit and TC-SoftPred (lambda=1.5).", "",
        "See five_task_summary.csv, history_validation, history_groups.csv and both selection locks.",
        "Test is confirmation on an already-used split, not a pristine holdout; it never changes selection.",
        f"New test evaluation performed: {test_status['test_accessed']}."]
    (target / "README.md").write_text("\n".join(lines) + "\n")
