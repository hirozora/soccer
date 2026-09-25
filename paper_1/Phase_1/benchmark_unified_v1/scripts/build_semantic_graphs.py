#!/usr/bin/env python
"""Build the isolated England semantic_v2 graph dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from football_benchmark.constants import GRAPH_ROOT, SEMANTIC_GRAPH_ROOT  # noqa: E402
from football_benchmark.semantic_graph import build_semantic_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=GRAPH_ROOT)
    parser.add_argument("--output-root", type=Path, default=SEMANTIC_GRAPH_ROOT)
    parser.add_argument("--competition", default="England")
    parser.add_argument("--limit-matches", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = build_semantic_dataset(
        source_root=args.source_root,
        output_root=args.output_root,
        competition=args.competition,
        overwrite=args.overwrite,
        limit_matches=args.limit_matches,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
