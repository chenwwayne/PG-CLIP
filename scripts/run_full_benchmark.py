#!/usr/bin/env python3
"""Run all directions, three seeds and five source-shot settings."""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


BENCHMARK_DIRECTIONS = {
    "amoled": ("active_to_cell", "cell_to_active"),
    "mvtec": ("carpet_to_grid", "grid_to_carpet"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    project = Path(__file__).resolve().parents[1]
    parser.add_argument("--project-root", type=Path, default=project)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--benchmark", choices=("amoled", "mvtec", "all"), default="all")
    parser.add_argument("--seeds", default="0,1,9")
    parser.add_argument("--shots", default="2,4,8,16,32")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    suites = tuple(BENCHMARK_DIRECTIONS) if args.benchmark == "all" else (args.benchmark,)
    directions = [direction for suite in suites for direction in BENCHMARK_DIRECTIONS[suite]]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    shots = [int(value) for value in args.shots.split(",") if value.strip()]
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        parser.error("--gpus must contain at least one GPU index")

    jobs = []
    settings = [
        (direction, seed, shot)
        for direction in directions
        for seed in seeds
        for shot in shots
    ]
    for index, (direction, seed, shot) in enumerate(settings):
        command = [
            sys.executable,
            str(args.project_root / "scripts" / "run_pgclip_protocol.py"),
            "--project-root", str(args.project_root),
            "--data-root", str(args.data_root),
            "--direction", direction,
            "--seed", str(seed),
            "--shot", str(shot),
            "--gpu", str(gpus[index % len(gpus)]),
        ]
        if args.dry_run:
            command.append("--dry-run")
        jobs.append(command)

    def run(command: list[str]) -> None:
        print("$", " ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=args.project_root, check=True)

    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        list(executor.map(run, jobs))


if __name__ == "__main__":
    main()
