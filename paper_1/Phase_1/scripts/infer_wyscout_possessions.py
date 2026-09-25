#!/usr/bin/env python3
"""Infer causal possession states from normalized Wyscout event tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PHASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_ROOT / "src"))

from football_hgt.event_tables import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT  # noqa: E402
from football_hgt.possession import (  # noqa: E402
    DEFAULT_POSSESSION_ROOT,
    DEFAULT_RULES_PATH,
    infer_possessions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_POSSESSION_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES_PATH)
    parser.add_argument("--competition", default="England")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = infer_possessions(
        event_root=args.event_root,
        output_root=args.output_root,
        data_root=args.data_root,
        competition=args.competition,
        rules_path=args.rules,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest["totals"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

