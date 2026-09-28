#!/usr/bin/env python3
"""Aggregate per-run metric JSON files into CSV and JSON summaries."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


METRICS = ("image_auroc", "image_ap", "pixel_auroc", "pixel_ap")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for manifest_path in sorted(args.results_root.rglob("manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_root = manifest_path.parent
        for method in manifest["methods"]:
            metrics_path = run_root / method / "target_evaluation" / "metrics.json"
            if not metrics_path.is_file():
                continue
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            rows.append({
                "direction": manifest["direction"], "seed": manifest["seed"],
                "shot": manifest["shot"], "method": method,
                **{metric: float(metrics[metric]) for metric in METRICS},
            })

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["direction"], row["shot"], row["method"])].append(row)
    summary = []
    for (direction, shot, method), group in sorted(grouped.items()):
        summary.append({
            "direction": direction, "shot": shot, "method": method,
            "seeds": len(group),
            **{metric: sum(row[metric] for row in group) / len(group) for metric in METRICS},
        })

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_prefix.with_suffix(".csv")
    json_path = args.output_prefix.with_suffix(".json")
    fields = ("direction", "shot", "method", "seeds", *METRICS)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
