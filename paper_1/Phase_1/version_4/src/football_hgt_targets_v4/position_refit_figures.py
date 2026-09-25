"""Standalone curves and artifact manifest for Position Head confirmation."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .constants import CONFIRMATION_SEEDS, VERSION_ROOT
from .position_head_refit import ROOT, run_dir, sha256, write_json


def finalize_artifacts(root: Path = ROOT) -> None:
    output = root / "report"
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    for axis, seed in zip(axes, CONFIRMATION_SEEDS):
        history = json.loads((run_dir(seed, root) / "history.json").read_text())
        result = json.loads((run_dir(seed, root) / "result.json").read_text())
        epochs = [row["epoch"] for row in history]
        values = [row["validation"]["mean_distance_m"] for row in history]
        axis.plot(epochs, values, marker=".", color="#16735d", label="Position refit")
        axis.axhline(values[0], linestyle="--", color="#666666", label="Original")
        best = result["best_epoch"]
        axis.scatter([best], [values[best]], color="#b74747", zorder=3, label="Selected")
        axis.set(title=f"Seed {seed}", xlabel="Epoch", ylabel="Validation mean distance (m)")
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    figure.savefig(output / "validation_curves.png", dpi=160)
    plt.close(figure)
    paths = [Path(__file__).with_name(name) for name in (
        "position_head_refit.py", "position_refit_verification.py", "position_refit_figures.py")]
    paths += [VERSION_ROOT / "scripts/run_position_head_refit.py", VERSION_ROOT / "tests/test_position_head_refit.py"]
    final = json.loads((output / "report.json").read_text())
    write_json(root / "manifest.json", {
        "status": "completed", "validation_selected": final["validation_lock"]["selected_method"],
        "test_evaluated": not final["test_skipped"],
        "test_confirmed": final["test_confirmation"]["effective"] if final["test_confirmation"] else None,
        "source_files": {str(path): sha256(path) for path in paths},
        "source_artifacts_unchanged": final["source_artifacts_unchanged"],
        "report_sha256": sha256(output / "report.json"),
        "training": {str(seed): {key: json.loads((run_dir(seed, root) / "result.json").read_text())[key]
                                  for key in ("best_epoch", "completed_epoch", "stopped_early", "initial_head_sha256")}
                     for seed in CONFIRMATION_SEEDS},
    })
