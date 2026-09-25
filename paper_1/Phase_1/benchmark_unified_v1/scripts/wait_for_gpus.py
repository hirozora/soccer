#!/usr/bin/env python
"""Wait until requested GPUs are idle enough for benchmark jobs."""

from __future__ import annotations

import argparse
import subprocess
import time


def query() -> dict[int, tuple[int, int, int]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    result = {}
    for line in output.splitlines():
        index, total, free, utilization = (int(value.strip()) for value in line.split(","))
        result[index] = (total, free, utilization)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--minimum-free-mib", type=int, default=20_000)
    parser.add_argument("--maximum-utilization", type=int, default=20)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    while True:
        state = query()
        ready = all(
            state[index][1] >= args.minimum_free_mib
            and state[index][2] <= args.maximum_utilization
            for index in args.gpus
        )
        print(
            " ".join(
                f"gpu{index}:free={state[index][1]}MiB,util={state[index][2]}%"
                for index in args.gpus
            ),
            flush=True,
        )
        if ready:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
