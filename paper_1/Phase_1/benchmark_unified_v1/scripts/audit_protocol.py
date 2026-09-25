#!/usr/bin/env python
"""Audit all England train/validation/test transitions and mapping invariants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from football_benchmark.audit import audit_protocol  # noqa: E402
from football_benchmark.constants import DEFAULT_ARTIFACT_PATH  # noqa: E402
from football_benchmark.protocol import ProtocolArtifacts  # noqa: E402
from football_benchmark.sampling import TargetSamplePlan  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--sample-plan-path", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/audit.json")
    args = parser.parse_args()
    sample_plan = (
        TargetSamplePlan.load(args.sample_plan_path) if args.sample_plan_path else None
    )
    report = audit_protocol(ProtocolArtifacts.load(args.artifact_path), sample_plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
