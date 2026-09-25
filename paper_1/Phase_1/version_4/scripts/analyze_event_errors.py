#!/usr/bin/env python
"""Validation-only Event error audit; no model fitting or test reads."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(VERSION_ROOT / "src"), str(VERSION_ROOT.parent / "benchmark_unified_v1/src")]

import numpy as np
import pandas as pd
import torch

from football_hgt_targets_v4.constants import (
    CONFIRMATION_SEEDS, EXPERIMENT_ROOT, PHASE_ROOT, POSSESSION_GRAPH_ROOT,
    RAW_EVENT_NAMES, SAMPLE_PLAN, SPLIT_PATH,
)
from football_hgt_targets_v4.event_posterior_integration import confusion
from football_hgt_targets_v4.position_head_refit import sha256, write_json

ROOT = EXPERIMENT_ROOT / "event_error_audit_v1"
PRED_ROOT = EXPERIMENT_ROOT / "event_posterior_integration_v1/training/null"


def graph_rows(matches, base_vocabulary, vocabulary, subevent_names):
    rows, hashes = [], {}
    for match in matches:
        path = POSSESSION_GRAPH_ROOT / "graphs/England" / f"{match}.pt"
        graph = torch.load(path, map_location="cpu", weights_only=False)
        hashes[str(path)] = sha256(path)
        event = graph["node_stores"]["event"]
        ids = event["raw_id"].numpy()
        if int(graph["match_id"]) != match or len(ids) != len(np.unique(ids)):
            raise RuntimeError("Raw Event IDs do not identify this match")
        raw_types = np.array([base_vocabulary["event_type_ids"][i] - 1 for i in event["event_type_index"].tolist()])
        subevents = np.array([subevent_names.get(base_vocabulary["subevent_type_ids"][i], "UNKNOWN")
                             for i in event["subevent_type_index"].tolist()])
        period = event["period_index"].numpy()
        n = len(ids) - 1
        if (not torch.equal(graph["targets"]["event_type_index"][:n], event["event_type_index"][1:])
                or not bool(graph["targets"]["mask"][:n].all())):
            raise RuntimeError("Graph targets are not the immediately adjacent next events")
        frame = pd.DataFrame({"sample_id": [f"{match}:{i}" for i in range(n)], "match_id": match,
            "current_event_index": np.arange(n), "anchor_event_uid": ids[:-1], "target_event_uid": ids[1:],
            "anchor_event": raw_types[:-1], "event_true": raw_types[1:],
            "anchor_subevent": subevents[:-1], "target_subevent": subevents[1:],
            "anchor_period": period[:-1], "next_period_break": period[:-1] != period[1:]})
        for name, field in (("control_state", "control_state_after_index"), ("event_role", "event_role_index"),
                            ("actor_relation_to_owner", "actor_relation_to_owner_index"),
                            ("candidate_status", "candidate_status_after_index")):
            frame[name] = event[field][:n].numpy()
            frame[f"{name}_name"] = frame[name].map(dict(enumerate(vocabulary["values"][name])))
        frame["switch_confirmed"] = event["switch_confirmed"][:n].numpy()
        frame["anchor_event_name"] = frame.anchor_event.map(dict(enumerate(RAW_EVENT_NAMES)))
        if not np.array_equal(event["period_index"][:-1].numpy() != event["period_index"][1:].numpy(), frame.next_period_break):
            raise RuntimeError("Phase alignment mismatch")
        rows.append(frame)
    return pd.concat(rows, ignore_index=True), hashes


def conditional_statistics(frame, population):
    output = []
    for fields in (("anchor_event_name",), ("control_state_name",), ("event_role_name",),
                   ("anchor_event_name", "control_state_name")):
        for group, chunk in frame.groupby(list(fields), dropna=False, sort=True):
            group = group if isinstance(group, tuple) else (group,)
            counts = np.bincount(chunk.event_true.to_numpy(dtype=int), minlength=10)
            probabilities = counts / counts.sum()
            entropy = -(probabilities[probabilities > 0] * np.log2(probabilities[probabilities > 0])).sum()
            for label, count in enumerate(counts):
                output.append({"population": population, "group_fields": "+".join(fields),
                    "group": " | ".join(map(str, group)), "next_event": RAW_EVENT_NAMES[label],
                    "samples": len(chunk), "next_count": int(count), "next_probability": probabilities[label],
                    "next_entropy_bits": entropy})
    return output


def main():
    torch.set_num_threads(1)
    ROOT.mkdir(parents=True, exist_ok=True)
    paths = [PRED_ROOT / f"seed{seed}/validation_original.parquet" for seed in CONFIRMATION_SEEDS]
    frames = [pd.read_parquet(path) for path in paths]
    keys = ["sample_id", "match_id", "current_event_index", "event_true", "player_mask", "event_role", "control_state", "switch_confirmed"]
    for other in frames[1:]:
        if not frames[0][keys].equals(other[keys]):
            raise RuntimeError("Seed predictions are not exactly aligned")
    if len(frames[0]) != 7296 or frames[0].sample_id.duplicated().any():
        raise RuntimeError("Unexpected validation sample population")
    split = pd.read_csv(SPLIT_PATH)
    split = split[split.competition_slug == "England"]
    train_matches = split.loc[split.split == "train", "match_id"].astype(int).tolist()
    val_matches = sorted(frames[0].match_id.unique())
    if set(val_matches) != set(split.loc[split.split == "validation", "match_id"]):
        raise RuntimeError("Validation split differs")
    vocabulary_path = POSSESSION_GRAPH_ROOT / "metadata/vocabularies.json"
    base_vocabulary_path = POSSESSION_GRAPH_ROOT.parent / "v1/metadata/vocabularies.json"
    mapping_path = PHASE_ROOT / "data/whyscout/raw/mappings/eventid2name.csv"
    vocabulary = json.loads(vocabulary_path.read_text())
    base_vocabulary = json.loads(base_vocabulary_path.read_text())
    mapping = pd.read_csv(mapping_path)
    subevent_names = {str(int(row.subevent)): row.subevent_label for row in mapping.itertuples()}
    validation, graph_hashes = graph_rows(val_matches, base_vocabulary, vocabulary, subevent_names)
    enriched = []
    for frame in frames:
        added = frame.merge(validation, on=["sample_id", "match_id", "current_event_index", "event_true",
            "event_role", "control_state", "switch_confirmed"], how="left", validate="one_to_one", indicator=True)
        if not (added["_merge"] == "both").all():
            raise RuntimeError("Prediction labels or causal state mismatch source graph")
        logits = added[[f"logit_{i}" for i in range(10)]].to_numpy()
        posterior = torch.tensor(logits).softmax(-1).numpy()
        order = np.argsort(-logits, axis=1, kind="stable")
        added["true_rank"] = np.argmax(order == added.event_true.to_numpy()[:, None], axis=1) + 1
        added["true_probability"] = posterior[np.arange(len(added)), added.event_true.to_numpy()]
        added["predicted_probability"] = posterior.max(1)
        added["correct"] = added.event_true == added.event_pred
        enriched.append(added.drop(columns="_merge"))
    data = pd.concat(enriched, ignore_index=True)
    data.to_parquet(ROOT / "validation_audit_samples.parquet", index=False)
    matrices = np.stack([confusion(f.event_true, f.event_pred) for f in frames])
    mean_cm = matrices.mean(0)
    support = matrices[0].sum(1)
    total_errors = float(mean_cm.sum() - np.trace(mean_cm))
    pd.DataFrame(mean_cm, index=RAW_EVENT_NAMES, columns=RAW_EVENT_NAMES).to_csv(ROOT / "confusion_mean.csv")
    normalized = mean_cm / support[:, None]
    pd.DataFrame(normalized, index=RAW_EVENT_NAMES, columns=RAW_EVENT_NAMES).to_csv(ROOT / "confusion_row_normalized.csv")
    pairs, class_rows = [], []
    for true in range(10):
        for predicted in range(10):
            if true != predicted and mean_cm[true, predicted] > 0:
                pairs.append({"true_event": RAW_EVENT_NAMES[true], "predicted_event": RAW_EVENT_NAMES[predicted],
                    "mean_errors": mean_cm[true, predicted], "within_true_class_rate": normalized[true, predicted],
                    "share_of_all_errors": mean_cm[true, predicted] / total_errors,
                    "seed_counts": "/".join(map(str, matrices[:, true, predicted]))})
        chunk = data[data.event_true == true]
        precisions, recalls, f1s = [], [], []
        for cm in matrices:
            tp, actual, predicted = cm[true, true], cm[true].sum(), cm[:, true].sum()
            precisions.append(tp / predicted if predicted else 0.)
            recalls.append(tp / actual)
            f1s.append(2 * tp / (actual + predicted) if actual + predicted else 0.)
        class_rows.append({"event": RAW_EVENT_NAMES[true], "unique_samples": int(support[true]),
            "predicted_count_mean": mean_cm[:, true].sum(), "precision": np.mean(precisions),
            "recall": np.mean(recalls), "f1": np.mean(f1s), "f1_std": np.std(f1s, ddof=1),
            "mean_errors": support[true] - mean_cm[true, true], "top3_recall": (chunk.true_rank <= 3).mean(),
            "mean_true_rank": float(chunk.true_rank.mean()), "mean_true_probability": float(chunk.true_probability.mean())})
    pair_table = pd.DataFrame(pairs).sort_values("mean_errors", ascending=False)
    pair_table.to_csv(ROOT / "top_confusions.csv", index=False)
    pd.DataFrame(class_rows).to_csv(ROOT / "per_class.csv", index=False)
    context_rows = []
    for field in ("anchor_event_name", "anchor_subevent", "control_state_name", "event_role_name", "switch_confirmed"):
        for (value, true), chunk in data.groupby([field, "event_true"], dropna=False):
            n = len(chunk) / len(CONFIRMATION_SEEDS)
            errors = chunk[~chunk.correct]
            for predicted, bad in errors.groupby("event_pred"):
                context_rows.append({"context_field": field, "context": value, "true_event": RAW_EVENT_NAMES[true],
                    "predicted_event": RAW_EVENT_NAMES[predicted], "unique_class_samples_in_context": n,
                    "mean_errors": len(bad) / 3, "within_context_class_error_rate": len(bad) / len(chunk),
                    "context_class_recall": chunk.correct.mean()})
    pd.DataFrame(context_rows).sort_values("mean_errors", ascending=False).to_csv(ROOT / "confusions_by_anchor_context.csv", index=False)
    subevents = data.groupby(["event_true", "target_subevent", "event_pred"]).size().rename("seed_predictions").reset_index()
    subevents["mean_predictions"] = subevents.seed_predictions / 3
    subevents["true_event_name"] = subevents.event_true.map(dict(enumerate(RAW_EVENT_NAMES)))
    subevents["predicted_event_name"] = subevents.event_pred.map(dict(enumerate(RAW_EVENT_NAMES)))
    subevents.to_csv(ROOT / "confusions_by_target_subevent_audit_only.csv", index=False)
    prediction_matrix = np.stack([frame.event_pred.to_numpy() for frame in frames])
    truth = frames[0].event_true.to_numpy()
    errors_per_sample = (prediction_matrix != truth).sum(0)
    same_wrong = (errors_per_sample == 3) & (prediction_matrix == prediction_matrix[0]).all(0)
    consensus = {"wrong_seed_count_" + str(k): int((errors_per_sample == k).sum()) for k in range(4)}
    consensus["same_wrong_class_all_three"] = int(same_wrong.sum())
    # Training transition frequencies are descriptive priors, not fitted predictions.
    training, training_hashes = graph_rows(train_matches, base_vocabulary, vocabulary, subevent_names)
    graph_hashes.update(training_hashes)
    if len(training) != 449025:
        raise RuntimeError("Full training transition count changed")
    plan = json.loads(SAMPLE_PLAN.read_text())["selections"]["train"]
    sample_ids = {f"{m}:{a}" for m, indices in plan.items() for a in indices}
    sampled = training[training.sample_id.isin(sample_ids)]
    if len(sampled) != 34048:
        raise RuntimeError("Sampled training count changed")
    pd.DataFrame(conditional_statistics(sampled, "sampled_train_34048") +
                 conditional_statistics(training, "full_train_449025")).to_csv(ROOT / "training_transition_statistics.csv", index=False)
    train_counts = pd.DataFrame({"event": RAW_EVENT_NAMES,
        "sampled_train": np.bincount(sampled.event_true, minlength=10),
        "full_train": np.bincount(training.event_true, minlength=10), "validation": support})
    train_counts.to_csv(ROOT / "class_support.csv", index=False)
    source_paths = [*paths, vocabulary_path, base_vocabulary_path, mapping_path, SPLIT_PATH, SAMPLE_PLAN]
    source_hashes = {str(path): sha256(path) for path in source_paths}
    source_hashes.update(graph_hashes)
    summary = {"validation_unique_samples": 7296, "seeds": list(CONFIRMATION_SEEDS),
        "validation_matches": len(val_matches), "train_matches": len(train_matches),
        "mean_errors": total_errors, "top_two_error_share": float(pair_table.head(2).share_of_all_errors.sum()),
        "error_agreement": consensus, "per_class": class_rows,
        "test_accessed": False, "training_performed": False,
        "warning": "Descriptive validation audit, not a new confirmatory model comparison; seed repetitions are not independent events."}
    write_json(ROOT / "summary.json", summary)
    write_json(ROOT / "manifest.json", {"source_hashes": source_hashes, "code_sha256": sha256(Path(__file__)),
        "test_accessed": False, "model_fitting": False, "graph_identity": "graph Event UID/order and adjacent targets; causal anchor states checked against prediction metadata"})
    lines = ["# Original Event Head Error Audit", "",
        "Validation only: 7,296 unique next-event samples from 57 England matches, evaluated by three fixed seeds.",
        "Counts below are mean prediction counts across seeds, not 21,888 independent events.",
        f"Mean errors per seed: {total_errors:.1f}. Two largest directional confusions account for {summary['top_two_error_share']:.1%} of errors.",
        "", "| True next Event | Predicted as | Mean errors | Fraction of true class |",
        "|---|---|---:|---:|"]
    for row in pair_table.head(12).itertuples():
        lines.append(f"| {row.true_event} | {row.predicted_event} | {row.mean_errors:.1f} | {row.within_true_class_rate:.1%} |")
    lines += ["", "| Class | Unique samples | Precision | Recall | F1 | Top-3 recall |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in class_rows:
        lines.append(f"| {row['event']} | {row['unique_samples']} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | {row['top3_recall']:.1%} |")
    lines += ["", "## Interpretation Boundaries", "",
        "- Most raw errors are concentrated at the Duel/Others versus Pass boundary; Offside is never the top prediction.",
        "- True-class-conditioned recall is diagnostic, not the deployment probability of that class in a state.",
        "- Training transition probabilities are reported separately for the actual 34,048 training samples and full 449,025 training transitions.",
        "- State and transition statistics are descriptive. They do not establish that a transition prior or hierarchical head will improve prediction.",
        "- Target subevent information is audit-only, never an input feature. UNKNOWN means the stored graph has no subevent label.",
        "- Source Event UIDs, adjacency targets and anchor states were checked against unchanged V3 graphs. No test predictions or test graphs were loaded.",
        "", "## Artifacts", "",
        "![Row-normalized confusion](confusion_heatmap.png)", "",
        "See per_class.csv, top_confusions.csv, confusions_by_anchor_context.csv, training_transition_statistics.csv, class_support.csv and summary.json."]
    (ROOT / "README.md").write_text("\n".join(lines) + "\n")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/football-event-audit-mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 9), constrained_layout=True)
    im = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    labels = [name.replace("Goalkeeper leaving line", "Goalkeeper exit").replace("Others on the ball", "Others") for name in RAW_EVENT_NAMES]
    ax.set_xticks(range(10), labels, rotation=45, ha="right")
    ax.set_yticks(range(10), labels)
    ax.set(xlabel="Predicted next Event", ylabel="True next Event", title="Original Event Head: validation confusion (three-seed mean)")
    for i in range(10):
        for j in range(10):
            if normalized[i, j] > 0:
                ax.text(j, i, f"{normalized[i,j]:.0%}", ha="center", va="center", color="white" if normalized[i,j] > .5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, label="Within-class fraction")
    fig.savefig(ROOT / "confusion_heatmap.png", dpi=140)
    plt.close(fig)
    print(json.dumps(summary, indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()
