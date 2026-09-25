#!/usr/bin/env python
"""Build the independent England Semantic V3 Possession graph dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from football_benchmark.constants import (  # noqa: E402
    POSSESSION_GRAPH_ROOT,
    POSSESSION_V2_ROOT,
    SEMANTIC_GRAPH_ROOT,
)
from football_benchmark.possession_graph import build_possession_graph_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-root", type=Path, default=SEMANTIC_GRAPH_ROOT)
    parser.add_argument("--possession-root", type=Path, default=POSSESSION_V2_ROOT)
    parser.add_argument("--output-root", type=Path, default=POSSESSION_GRAPH_ROOT)
    parser.add_argument("--competition", default="England")
    parser.add_argument("--limit-matches", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = build_possession_graph_dataset(
        semantic_root=args.semantic_root,
        possession_root=args.possession_root,
        output_root=args.output_root,
        competition=args.competition,
        overwrite=args.overwrite,
        limit_matches=args.limit_matches,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
