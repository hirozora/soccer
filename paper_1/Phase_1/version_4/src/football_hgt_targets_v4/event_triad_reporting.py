"""Full-ten-class and anchor-state reporting for the triad experiment."""

import json
import numpy as np
import pandas as pd

from .constants import CONFIRMATION_SEEDS, RAW_EVENT_NAMES
from .event_triad import ROOT, compare, read_validation, require_lock
from .event_posterior_integration import confusion, cm_scores
from .position_head_refit import write_json


def summarize_frames(frames, split, output):
    metrics, classes, pairs, groups = [], [], [], []
    for mode, frame in frames.items():
        for seed, part in frame.groupby("seed", sort=True):
            cm = confusion(part.event_true, part.event_pred)
            f1, acc = cm_scores(cm)
            metrics.append({"split": split, "mode": mode, "seed": seed, "macro_f1": f1, "accuracy": acc})
            pd.DataFrame(cm, index=RAW_EVENT_NAMES, columns=RAW_EVENT_NAMES).to_csv(output / f"{split}_{mode}_{seed}_confusion.csv")
            for k, name in enumerate(RAW_EVENT_NAMES):
                tp, actual, pred = cm[k, k], cm[k].sum(), cm[:, k].sum()
                classes.append({"split": split, "mode": mode, "seed": seed, "class": name, "support": actual,
                    "predicted_count": pred, "precision": tp / max(pred, 1), "recall": tp / max(actual, 1),
                    "f1": 2 * tp / max(actual + pred, 1)})
            for a, b in ((0, 7), (6, 7), (7, 0), (7, 6)):
                pairs.append({"split": split, "mode": mode, "seed": seed, "true": RAW_EVENT_NAMES[a],
                              "predicted": RAW_EVENT_NAMES[b], "errors": cm[a, b], "true_class_rate": cm[a, b] / max(cm[a].sum(), 1)})
            for fields in (("anchor_type",), ("control_state",), ("anchor_type", "control_state"), ("match_id",)):
                for key, chunk in part.groupby(list(fields), sort=True):
                    group_cm = confusion(chunk.event_true, chunk.event_pred)
                    gf1, ga = cm_scores(group_cm)
                    groups.append({"split": split, "mode": mode, "seed": seed, "fields": "+".join(fields),
                        "group": str(key), "samples": len(chunk), "small_group": len(chunk) < 30,
                        "macro_f1": gf1, "accuracy": ga, "duel_to_pass": group_cm[0, 7],
                        "others_to_pass": group_cm[6, 7], "pass_to_duel": group_cm[7, 0], "pass_to_others": group_cm[7, 6]})
    for name, rows in (("per_seed", metrics), ("per_class", classes), ("directional_errors", pairs), ("groups", groups)):
        pd.DataFrame(rows).to_csv(output / f"{split}_{name}.csv", index=False)
    table = pd.DataFrame(metrics)
    return {mode: {metric: {"mean": float(chunk[metric].mean()), "std": float(chunk[metric].std(ddof=1))}
                   for metric in ("macro_f1", "accuracy")} for mode, chunk in table.groupby("mode")}


def report(root=ROOT):
    lock = json.loads((root / "selection/event_triad_lock.json").read_text())
    output = root / "report"
    output.mkdir(parents=True, exist_ok=True)
    frames = {"original": pd.read_parquet(root / "calibration/validation_original.parquet"),
              "calibration_oof": pd.read_parquet(root / "calibration/validation_oof.parquet")}
    if not json.loads((root / "calibration/decision.json").read_text())["passed"]:
        frames.update({m: read_validation(root, m) for m in ("context", "state")})
    summary = {"validation": summarize_frames(frames, "validation", output), "lock": lock}
    training, efficiency = [], []
    for path in sorted((root / "training").glob("*/seed*/result.json")):
        run = json.loads(path.read_text())
        training.append({"mode": run["config"]["mode"], "seed": run["config"]["seed"],
                         "best_epoch": run["best_epoch"], "completed_epoch": run["completed_epoch"],
                         "elapsed_seconds": run["elapsed_seconds"], "trainable_parameters": 2627})
    for path in sorted((root / "verification").glob("*.json")):
        verified = json.loads(path.read_text())
        timing = verified["descriptive_model_timings_seconds"]
        efficiency.append({"mode": verified["mode"], "seed": verified["seed"], "verified": verified["passed"],
            "original_seconds": float(np.mean(timing["original"])),
            "candidate_seconds": float(np.mean(timing["candidate"])),
            "relative_descriptive_time": float(np.sum(timing["candidate"]) / np.sum(timing["original"])),
            "other_four_max_error": max(verified["unaffected_output_max_errors"].values()),
            **{f"calls_{k}": v for k, v in verified["layer_calls"].items()}})
    pd.DataFrame(training).to_csv(output / "training_summary.csv", index=False)
    pd.DataFrame(efficiency).to_csv(output / "online_checks_and_descriptive_cost.csv", index=False)
    summary["training"] = training
    summary["online_checks"] = efficiency
    if lock["passed"]:
        require_lock(root)
        test_frames = {mode: pd.concat([pd.read_parquet(root / "test" / f"seed{seed}/test_{mode}.parquet")
                                        for seed in CONFIRMATION_SEEDS], ignore_index=True) for mode in lock["test_modes"]}
        summary["test"] = summarize_frames(test_frames, "test", output)
        selected = lock["selected_method"]
        summary["test_comparisons"] = {f"{selected}_minus_original": compare(test_frames["original"], test_frames[selected])}
        if selected == "state":
            summary["test_comparisons"]["state_minus_context"] = compare(test_frames["context"], test_frames["state"])
        summary["test_confirmed"] = all(v["effective"] for v in summary["test_comparisons"].values())
    write_json(output / "summary.json", summary)
    lines = ["# Event Triad Refinement", "", f"Validation-locked method: **{lock['selected_method']}**.",
             "Calibration is evaluated out of fold; its final bias is fitted only after that decision.",
             "", "| Split / Method | Accuracy | Macro-F1 |", "|---|---:|---:|"]
    for split in ("validation", "test"):
        for mode, scores in summary.get(split, {}).items():
            a, f = scores["accuracy"], scores["macro_f1"]
            lines.append(f"| {split} / {mode} | {a['mean']:.6f} +/- {a['std']:.6f} | {f['mean']:.6f} +/- {f['std']:.6f} |")
    lines += ["", "## Registered Comparisons", ""]
    for name, value in lock["comparisons"].items():
        lines.append(f"- {name}: F1 {value['macro_f1_gain']:+.6f}, CI {value['ci95']}; Accuracy {value['accuracy_gain']:+.6f}; effective={value['effective']}.")
    directions = pd.read_csv(output / "validation_directional_errors.csv")
    means = directions.groupby(["mode", "true", "predicted"], sort=True).errors.mean()
    lines += ["", "## Bidirectional Errors", "", "Three-seed mean counts on the same 7,296 validation events.", "",
              "| Method | Duel to Pass | Others to Pass | Pass to Duel | Pass to Others |", "|---|---:|---:|---:|---:|"]
    for mode in frames:
        values = [means[mode, a, b] for a, b in (("Duel", "Pass"), ("Others on the ball", "Pass"),
                                               ("Pass", "Duel"), ("Pass", "Others on the ball"))]
        lines.append(f"| {mode} | " + " | ".join(f"{v:.2f}" for v in values) + " |")
    lines += ["", (f"Test confirmation: {summary['test_confirmed']}; no method was reselected." if lock["passed"]
                    else "Validation admission failed. Original retained; no test data or predictions read."),
              "", "Other seven probabilities are preserved, but their argmax classifications may change.",
              "Online timings use two validation batches without a dedicated warmed benchmark; they are descriptive, not a speedup claim.",
              "Anchor-state analyses are descriptive; target subevents are not inputs.",
              "Previously seen test split: confirmation on an existing split, not a pristine holdout."]
    (output / "README.md").write_text("\n".join(lines) + "\n")
    return summary
