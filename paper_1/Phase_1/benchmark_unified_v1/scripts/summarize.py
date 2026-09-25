#!/usr/bin/env python
"""Summarize final runs and compute paired hierarchical bootstrap CIs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from football_benchmark.bootstrap import hierarchical_paired_bootstrap  # noqa: E402

BASELINES = {"seq2event": "seq2event", "unified_lem": "unified_lem", "nmstpp": "nmstpp"}


def flatten_result(path: Path) -> dict[str, object]:
    result = json.loads(path.read_text(encoding="utf-8"))
    test = result["test"]
    row: dict[str, object] = {
        "contract": result["config"]["contract"],
        "family": result["config"]["family"],
        "window_size": result["config"]["window_size"],
        "seed": result["config"]["seed"],
        "learning_rate": result["config"]["learning_rate"],
        "parameters": result["parameters"],
        "best_epoch": result["best_epoch"],
        "elapsed_seconds": result["elapsed_seconds"],
        "unified_event_loss_mode": result["config"].get(
            "unified_event_loss_mode", "legacy_balanced"
        ),
        "graph_variant": result["config"].get("graph_variant", "legacy"),
        "peak_cuda_memory_bytes": result.get("peak_cuda_memory_bytes", 0),
        "event_accuracy": test["event"]["accuracy"],
        "event_macro_f1": test["event"]["macro_f1"],
        "event_weighted_f1": test["event"]["weighted_f1"],
        "position_distance_mae_m": test["position"]["distance_mae_m"],
    }
    if test["time"] is not None:
        row["time_mae_seconds"] = test["time"]["mae_seconds"]
        row["time_rmse_seconds"] = test["time"]["rmse_seconds"]
    if "zone_accuracy" in test["position"]:
        row["zone_accuracy"] = test["position"]["zone_accuracy"]
        row["zone_macro_f1"] = test["position"]["zone_macro_f1"]
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--profile", choices=("full", "feasibility"), default="full")
    parser.add_argument(
        "--unified-repair",
        action="store_true",
        help="Replace only Unified LEM baseline runs with repaired results",
    )
    parser.add_argument(
        "--semantic-hgt",
        action="store_true",
        help="Replace every legacy HGT run with semantic_v2 HGT results",
    )
    args = parser.parse_args()
    final_root = (
        ROOT / "experiments/feasibility/final"
        if args.profile == "feasibility"
        else ROOT / "experiments/final"
    )
    paths = sorted(final_root.glob("**/result.json"))
    legacy_hgt_paths = [
        path
        for path in paths
        if json.loads(path.read_text(encoding="utf-8"))["config"]["family"] == "hgt"
    ]
    if args.unified_repair:
        if args.profile != "feasibility":
            parser.error("--unified-repair is only available for the feasibility profile")
        retained: list[Path] = []
        for path in paths:
            result = json.loads(path.read_text(encoding="utf-8"))
            config = result["config"]
            if not (
                config["contract"] == "unified_lem"
                and config["family"] == "unified_lem"
            ):
                retained.append(path)
        repair_paths = sorted(
            (
                ROOT
                / "experiments/feasibility/unified_lem_repair/final"
            ).glob("**/result.json")
        )
        if len(repair_paths) != 3:
            parser.error(f"Expected three repaired Unified results, found {len(repair_paths)}")
        paths = retained + repair_paths
    if args.semantic_hgt:
        if args.profile != "feasibility":
            parser.error("--semantic-hgt is only available for the feasibility profile")
        retained = []
        for path in paths:
            result = json.loads(path.read_text(encoding="utf-8"))
            if result["config"]["family"] != "hgt":
                retained.append(path)
        semantic_paths = sorted(
            (ROOT / "experiments/feasibility/semantic_hgt_v2/final").glob(
                "**/result.json"
            )
        )
        if len(semantic_paths) != 9:
            parser.error(f"Expected nine semantic HGT results, found {len(semantic_paths)}")
        paths = retained + semantic_paths
    if not paths:
        parser.error("No final result.json files found")
    runs = pd.DataFrame([flatten_result(path) for path in paths])
    group_columns = ["contract", "family", "graph_variant", "window_size"]
    numeric = [column for column in runs.select_dtypes("number") if column != "seed"]
    summary = runs.groupby(group_columns)[numeric].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(value for value in column if value).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    output = (
        ROOT
        / (
            "experiments/feasibility/summary_semantic_v2"
            if args.semantic_hgt
            else (
                "experiments/feasibility/summary_repaired"
                if args.unified_repair
                else "experiments/feasibility/summary"
            )
        )
        if args.profile == "feasibility"
        else ROOT / "experiments/summary"
    )
    output.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output / "runs.csv", index=False)
    summary.to_csv(output / "mean_std.csv", index=False)

    per_class_rows: list[dict[str, object]] = []
    for path in paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        for label, values in result["test"]["event"]["per_class"].items():
            per_class_rows.append(
                {
                    "contract": result["config"]["contract"],
                    "family": result["config"]["family"],
                    "graph_variant": result["config"].get("graph_variant", "legacy"),
                    "window_size": result["config"]["window_size"],
                    "seed": result["config"]["seed"],
                    "label": label,
                    **values,
                }
            )
    per_class = pd.DataFrame(per_class_rows)
    per_class.to_csv(output / "per_class_runs.csv", index=False)
    per_class_summary = (
        per_class.groupby(
            ["contract", "family", "graph_variant", "window_size", "label"]
        )[
            ["precision", "recall", "f1", "support"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    per_class_summary.columns = [
        "_".join(value for value in column if value).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in per_class_summary.columns
    ]
    per_class_summary.to_csv(output / "per_class_mean_std.csv", index=False)

    bootstrap: dict[str, object] = {}
    result_paths = {
        (
            json.loads(path.read_text(encoding="utf-8"))["config"]["contract"],
            json.loads(path.read_text(encoding="utf-8"))["config"]["family"],
            json.loads(path.read_text(encoding="utf-8"))["config"]["window_size"],
            json.loads(path.read_text(encoding="utf-8"))["config"]["seed"],
        ): path
        for path in paths
    }
    expected_seeds = 3 if args.profile == "feasibility" else 5
    for contract, baseline in BASELINES.items():
        windows = sorted(runs[runs.contract == contract].window_size.unique())
        for window in windows:
            hgt_seeds = sorted(
                runs[
                    (runs.contract == contract)
                    & (runs.family == "hgt")
                    & (runs.window_size == window)
                ].seed
            )
            baseline_seeds = sorted(
                runs[
                    (runs.contract == contract)
                    & (runs.family == baseline)
                    & (runs.window_size == window)
                ].seed
            )
            if hgt_seeds != baseline_seeds or len(hgt_seeds) != expected_seeds:
                raise RuntimeError(
                    f"Expected {expected_seeds} paired seeds for {contract}/K{window}; "
                    f"got HGT={hgt_seeds}, baseline={baseline_seeds}"
                )
            hgt_paths = [
                result_paths[(contract, "hgt", window, seed)].parent
                / "test_predictions.parquet"
                for seed in hgt_seeds
            ]
            baseline_paths = [
                result_paths[(contract, baseline, window, seed)].parent
                / "test_predictions.parquet"
                for seed in baseline_seeds
            ]
            bootstrap[f"{contract}/k{window}"] = hierarchical_paired_bootstrap(
                [pd.read_parquet(path).sort_values("sample_id").reset_index(drop=True) for path in hgt_paths],
                [pd.read_parquet(path).sort_values("sample_id").reset_index(drop=True) for path in baseline_paths],
                contract,
                replicates=args.replicates,
            )
    (output / "paired_bootstrap.json").write_text(
        json.dumps(bootstrap, indent=2), encoding="utf-8"
    )
    if args.semantic_hgt:
        legacy = pd.DataFrame([flatten_result(path) for path in legacy_hgt_paths])
        semantic = runs[runs.family == "hgt"].copy()
        comparison_metrics = [
            "event_accuracy",
            "event_macro_f1",
            "event_weighted_f1",
            "position_distance_mae_m",
            "time_mae_seconds",
            "time_rmse_seconds",
        ]
        comparison_rows: list[dict[str, object]] = []
        for contract in BASELINES:
            left = legacy[legacy.contract == contract].set_index("seed")
            right = semantic[semantic.contract == contract].set_index("seed")
            for metric in comparison_metrics:
                if metric not in left or metric not in right:
                    continue
                valid = left[metric].notna() & right[metric].notna()
                for seed in left.index[valid]:
                    comparison_rows.append(
                        {
                            "contract": contract,
                            "seed": int(seed),
                            "metric": metric,
                            "legacy": float(left.loc[seed, metric]),
                            "semantic_v2": float(right.loc[seed, metric]),
                            "delta_semantic_minus_legacy": float(
                                right.loc[seed, metric] - left.loc[seed, metric]
                            ),
                        }
                    )
        comparison = pd.DataFrame(comparison_rows)
        comparison.to_csv(output / "hgt_legacy_vs_semantic_runs.csv", index=False)
        comparison_summary = (
            comparison.groupby(["contract", "metric"])[
                ["legacy", "semantic_v2", "delta_semantic_minus_legacy"]
            ]
            .agg(["mean", "std"])
            .reset_index()
        )
        comparison_summary.columns = [
            "_".join(value for value in column if value).rstrip("_")
            if isinstance(column, tuple)
            else column
            for column in comparison_summary.columns
        ]
        comparison_summary.to_csv(
            output / "hgt_legacy_vs_semantic_mean_std.csv", index=False
        )

        compact_specs = (
            ("HGT", "unified_lem", "hgt"),
            ("Unified LEM", "unified_lem", "unified_lem"),
            ("Soccer-SEQ2Event", "seq2event", "seq2event"),
            ("NMSTPP", "nmstpp", "nmstpp"),
        )
        compact_rows = []
        for method, contract, family in compact_specs:
            selected = runs[(runs.contract == contract) & (runs.family == family)]
            compact_rows.append(
                {
                    "method": method,
                    "event_accuracy_mean": selected.event_accuracy.mean(),
                    "event_accuracy_std": selected.event_accuracy.std(),
                    "time_mae_seconds_mean": selected.time_mae_seconds.mean(),
                    "time_mae_seconds_std": selected.time_mae_seconds.std(),
                    "position_distance_mae_m_mean": selected.position_distance_mae_m.mean(),
                    "position_distance_mae_m_std": selected.position_distance_mae_m.std(),
                }
            )
        pd.DataFrame(compact_rows).to_csv(output / "compact_methods.csv", index=False)

    if args.unified_repair or args.semantic_hgt:
        def sha256(path: Path) -> str:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        retained_baselines = []
        for path in paths:
            result = json.loads(path.read_text(encoding="utf-8"))
            if result["config"]["family"] == "hgt":
                continue
            prediction_path = path.parent / "test_predictions.parquet"
            retained_baselines.append(
                {
                    "result": str(path.resolve()),
                    "result_sha256": sha256(path),
                    "predictions": str(prediction_path.resolve()),
                    "predictions_sha256": sha256(prediction_path),
                }
            )
        (output / "provenance.json").write_text(
            json.dumps(
                {
                    "base_results": str(ROOT / "experiments/feasibility/final"),
                    "unified_repair": (
                        str(ROOT / "experiments/feasibility/unified_lem_repair/final")
                        if args.unified_repair
                        else None
                    ),
                    "semantic_hgt_replacement": (
                        str(ROOT / "experiments/feasibility/semantic_hgt_v2/final")
                        if args.semantic_hgt
                        else None
                    ),
                    "retained_baselines": retained_baselines,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
