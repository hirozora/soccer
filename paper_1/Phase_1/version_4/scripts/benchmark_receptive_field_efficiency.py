#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "benchmark_unified_v1/src")]
from football_hgt_targets_v4.receptive_field_efficiency import benchmark_receptive_fields  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--batches", type=int, default=10)
args = parser.parse_args()
print(benchmark_receptive_fields(args.device, args.batches))
