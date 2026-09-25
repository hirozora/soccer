"""Descriptive all-event reporting for the locked Event integration experiment."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .constants import CONFIRMATION_SEEDS
from .event_posterior_integration import (
    ROOT, MODES, cm_scores, comparisons, confusion, read_predictions, require_test_lock, run_dir,
)
from .partial_sharing_study import training_dir as partial_train, test_dir as partial_test
from .position_head_refit import ROOT as POSITION_ROOT
from .position_head_refit import sha256, write_json
from .team_candidate_prior import load_team_candidate_cache, prediction_frame as actor_frame
from .team_candidate_prior_study import cache_path as actor_cache_path


def fixed_metrics(seed, split, event_frame, root):
    if split == "test":
        require_test_lock(root)
    partial = (partial_train(seed) / "validation_predictions_guarded_core.parquet" if split == "validation"
               else partial_test(seed) / "test_predictions.parquet")
    position = (POSITION_ROOT / "training" / f"seed{seed}" / "validation_refit.parquet" if split == "validation"
                else POSITION_ROOT / "test" / f"seed{seed}" / "test_refit.parquet")
    original, xy = pd.read_parquet(partial), pd.read_parquet(position)
    cache = load_team_candidate_cache(seed, split)
    actors = actor_frame(cache, "soft", lambda_value=1.5)
    ids = event_frame.sample_id.tolist()
    for frame in (original, xy, actors):
        if frame.sample_id.tolist() != ids:
            raise RuntimeError("Composed five-task reporting samples differ")
    active = xy.position_mask.to_numpy(dtype=bool)
    distance = np.linalg.norm((xy[["position_pred_x", "position_pred_y"]].to_numpy() - xy[["position_true_x", "position_true_y"]].to_numpy()) * [105, 68], axis=1)[active]
    ranks = actors.loc[actors.player_mask.astype(bool), "player_rank"].to_numpy()
    valid_time = original.time_mask.astype(bool)
    return {"time_mae": float((original.loc[valid_time, "time_pred"] - original.loc[valid_time, "time_true"]).abs().mean()),
        "position_mean_m": float(distance.mean()), "position_median_m": float(np.median(distance)),
        "team_accuracy": float((original.team_pred == original.team_true).mean()),
        "player_top1": float((ranks <= 1).mean()), "player_top3": float((ranks <= 3).mean()),
        "player_top5": float((ranks <= 5).mean()), "player_mrr": float((1 / ranks).mean()),
        "player_coverage": float(actors.player_mask.mean()),
        "sources": {str(p): sha256(p) for p in (partial, position, actor_cache_path(seed, split))}}


def report(root=ROOT):
    lock = json.loads((root / "selection/event_posterior_lock.json").read_text())
    output = root / "report"
    output.mkdir(parents=True, exist_ok=True)
    from .event_posterior_online import benchmark_online
    efficiency = []
    for seed in CONFIRMATION_SEEDS:
        path = root / "efficiency" / f"seed{seed}.json"
        efficiency.extend(json.loads(path.read_text()) if path.exists() else benchmark_online(seed, root))
    pd.DataFrame(efficiency).to_csv(output / "efficiency.csv", index=False)
    splits = ["validation"]
    if lock["passed"] and all((root / "test" / f"seed{s}" / "test_base_post.parquet").exists() for s in CONFIRMATION_SEEDS):
        require_test_lock(root)
        splits.append("test")
    results, grouped, classes, source_hashes = [], [], [], {}
    per_split = {}
    for split in splits:
        per_split[split] = comparisons(split, root) if split == "test" else lock["comparisons"]
        fixed = {}
        for mode in ("original", *MODES):
            frame = read_predictions(split, mode, root)
            for seed, group in frame.groupby("seed", sort=True):
                if seed not in fixed:
                    fixed[seed] = fixed_metrics(seed, split, group, root)
                    source_hashes.update(fixed[seed]["sources"])
                cm = confusion(group.event_true, group.event_pred)
                f1, accuracy = cm_scores(cm)
                results.append({"split": split, "mode": mode, "seed": seed, "samples": len(group),
                    "event_accuracy": float(accuracy), "event_macro_f1": float(f1),
                    **{k: v for k, v in fixed[seed].items() if k != "sources"}})
                pd.DataFrame(cm).to_csv(output / f"{split}_{mode}_seed{seed}_confusion.csv", index=False)
                for label in range(10):
                    tp, support, predicted = cm[label, label], cm[label].sum(), cm[:, label].sum()
                    precision = tp / predicted if predicted else 0.0
                    recall = tp / support if support else 0.0
                    classes.append({"split": split, "mode": mode, "seed": seed, "event": label,
                        "support": int(support), "precision": precision, "recall": recall,
                        "f1": 2 * tp / (support + predicted) if support + predicted else 0.0})
                for field in ("player_mask", "event_true", "control_state", "event_role", "switch_confirmed"):
                    for value, subset in group.groupby(field):
                        score, acc = cm_scores(confusion(subset.event_true, subset.event_pred))
                        grouped.append({"split": split, "mode": mode, "seed": seed, "group": field, "value": value,
                                        "samples": len(subset), "macro_f1_raw10": float(score), "accuracy": float(acc)})
    metrics = pd.DataFrame(results)
    metrics.to_csv(output / "five_task_metrics_by_seed.csv", index=False)
    pd.DataFrame(classes).to_csv(output / "event_per_class.csv", index=False)
    pd.DataFrame(grouped).to_csv(output / "grouped_metrics.csv", index=False)
    summary = metrics.groupby(["split", "mode"])[[c for c in metrics if c not in ("split", "mode", "seed", "samples")]].agg(["mean", "std"])
    summary.to_csv(output / "summary.csv")
    histories = []
    for seed in CONFIRMATION_SEEDS:
        for mode in MODES:
            directory = run_dir(seed, mode, root)
            run = json.loads((directory / "result.json").read_text())
            histories.append({"seed": seed, "mode": mode, "best_epoch": run["best_epoch"],
                              "epochs": run["completed_epoch"], "training_seconds": run["elapsed_seconds"]})
            for population in ("train", "validation"):
                source_hashes.update(run["provenance"][population]["sources"])
    for path, expected in source_hashes.items():
        if sha256(Path(path)) != expected:
            raise RuntimeError(f"An existing source changed: {path}")
    test_confirmed = (all(per_split["test"][key]["effective"] for key in ("base_post_minus_null", "base_post_minus_original"))
                      if "test" in splits else None)
    payload = {"lock": lock, "comparisons": per_split, "runs": histories, "efficiency": efficiency,
        "test_confirmed": test_confirmed, "existing_sources_unchanged": True,
        "source_hashes": source_hashes, "test_selection": "never reselect from test",
        "interpretation": "Player posterior conditioning must beat both frozen Original and capacity-matched Null"}
    write_json(output / "report.json", payload)
    lines = ["# Event Posterior Integration", "", "Primary population: all Raw-10 events, including unknown Player.", "",
        "| Split | Method | Event Accuracy | Macro-F1 |", "|---|---|---:|---:|"]
    for (split, mode), row in summary.iterrows():
        lines.append(f"| {split} | {mode} | {row[('event_accuracy', 'mean')]:.6f} +/- {row[('event_accuracy', 'std')]:.6f} | {row[('event_macro_f1', 'mean')]:.6f} +/- {row[('event_macro_f1', 'std')]:.6f} |")
    lines.extend(["", "| Validation comparison | Macro-F1 gain | 95% CI | Effective |",
                  "|---|---:|---|---|"])
    for comparison, item in lock["comparisons"].items():
        lines.append(f"| {comparison} | {item['macro_f1_gain']:+.6f} | [{item['ci95'][0]:+.6f}, {item['ci95'][1]:+.6f}] | {item['effective']} |")
    test_status = str(test_confirmed) if "test" in splits else "Not evaluated (validation gate)."
    lines.extend(["", f"Validation selected: {lock['selected_method']}.", f"Test confirmation: {test_status}",
        "", "Warmed CPU model-only timing and parameter counts: efficiency.csv. HGT runs once per branch (3 layer calls total).",
        "", "Position Refit and TC-SoftPred (lambda=1.5) are fixed; Event uses the unadjusted Player posterior.",
        "No new test evaluation is authorized when validation fails. Existing test results were previously visible; this is not a pristine holdout."])
    (output / "README.md").write_text("\n".join(lines) + "\n")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/football-event-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), constrained_layout=True)
    for ax, seed in zip(axes, CONFIRMATION_SEEDS):
        for mode in MODES:
            history = json.loads((run_dir(seed, mode, root) / "history.json").read_text())
            ax.plot([r["epoch"] for r in history], [r["validation"]["macro_f1"] for r in history], label=mode)
        ax.set(title=str(seed), xlabel="Epoch", ylabel="Validation Macro-F1")
        ax.legend()
    fig.savefig(output / "training_curves.png", dpi=140)
    plt.close(fig)
    version_root = Path(__file__).resolve().parents[2]
    files = [Path(__file__), Path(__file__).with_name("event_posterior_integration.py"), Path(__file__).with_name("event_posterior_online.py"),
        version_root / "scripts/run_event_posterior_integration.py", version_root / "tests/test_event_posterior_integration.py"]
    write_json(root / "manifest.json", {"status": "completed", "validation_passed": lock["passed"],
        "test_evaluated": "test" in splits, "test_confirmed": test_confirmed,
        "source_hashes": source_hashes, "code_hashes": {str(p): sha256(p) for p in files},
        "report_sha256": sha256(output / "report.json")})
    return payload
