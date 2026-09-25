#!/usr/bin/env python3
"""Build normalized, graph-independent Wyscout event tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PHASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_ROOT / "src"))

from football_hgt.event_tables import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    build_event_tables,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--competition", default="England")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = build_event_tables(
        args.data_root, args.output_root, args.competition, args.overwrite
    )
    print(json.dumps(manifest["totals"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

