#!/usr/bin/env python3
"""Build graph-ready causal Wyscout possession data V2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PHASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_ROOT / "src"))

from football_hgt.event_tables import DEFAULT_OUTPUT_ROOT  # noqa: E402
from football_hgt.possession_v2 import (  # noqa: E402
    DEFAULT_CANDIDATE_RULES_PATH_V2,
    DEFAULT_POSSESSION_ROOT_V1,
    DEFAULT_POSSESSION_ROOT_V2,
    DEFAULT_RULES_PATH_V2,
    infer_possessions_v2,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_POSSESSION_ROOT_V2)
    parser.add_argument("--v1-root", type=Path, default=DEFAULT_POSSESSION_ROOT_V1)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES_PATH_V2)
    parser.add_argument(
        "--candidate-rules", type=Path, default=DEFAULT_CANDIDATE_RULES_PATH_V2
    )
    parser.add_argument("--competition", default="England")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = infer_possessions_v2(
        event_root=args.event_root,
        output_root=args.output_root,
        v1_root=args.v1_root,
        competition=args.competition,
        rules_path=args.rules,
        candidate_rules_path=args.candidate_rules,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest["totals"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
