#!/usr/bin/env python3
"""Build deterministic carpet/grid cross-domain splits from MVTec-AD."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path


DIRECTIONS = (("carpet", "grid"), ("grid", "carpet"))
SHOTS = (2, 4, 8, 16, 32)


@dataclass(frozen=True)
class Item:
    image_path: str
    label: int
    mask_path: str
    class_name: str


def stable_seed(seed: int, *parts: str) -> int:
    payload = "|".join(map(str, (seed, *parts))).encode()
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def image_files(root: Path, status: str) -> list[str]:
    return sorted(
        path.relative_to(root).as_posix()
        for path in (root / "images" / status).glob("*.png")
    )


def make_items(root: Path, category: str, paths: list[str], label: int) -> list[Item]:
    items = []
    for relative in paths:
        mask = ""
        if label:
            mask = (
                root / "mask" / "anomaly" / Path(relative).name
            ).relative_to(root).as_posix()
        items.append(Item(relative, label, mask, category))
    return items


def write_jsonl(path: Path, items: list[Item]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(asdict(item), ensure_ascii=False) + "\n" for item in items),
        encoding="utf-8",
    )


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def build_split(
    data_root: Path,
    output_root: Path,
    source: str,
    target: str,
    seed: int,
    shot: int,
) -> None:
    source_root = data_root / source
    target_root = data_root / target
    normal = image_files(source_root, "normal")
    anomalous = image_files(source_root, "anomaly")
    random.Random(stable_seed(seed, source, "normal")).shuffle(normal)
    random.Random(stable_seed(seed, source, "anomaly")).shuffle(anomalous)
    if len(normal) < 16 or len(anomalous) < 16:
        raise ValueError(f"{source} requires at least 16 normal and 16 anomalous images")

    pool_normal, pool_anomalous = normal[:16], anomalous[:16]
    normal_count, anomalous_count = (shot + 1) // 2, shot // 2
    selected_normal = pool_normal[:normal_count]
    selected_anomalous = pool_anomalous[:anomalous_count]
    source_pool = make_items(source_root, source, pool_normal, 0) + make_items(
        source_root, source, pool_anomalous, 1
    )
    source_subset = make_items(source_root, source, selected_normal, 0) + make_items(
        source_root, source, selected_anomalous, 1
    )
    unused = make_items(source_root, source, pool_normal[normal_count:], 0) + make_items(
        source_root, source, pool_anomalous[anomalous_count:], 1
    )
    target_test = make_items(target_root, target, image_files(target_root, "normal"), 0)
    target_test += make_items(target_root, target, image_files(target_root, "anomaly"), 1)

    direction = f"{source}_to_{target}"
    destination = output_root / direction / f"seed_{seed}" / f"shot_{shot}"
    write_jsonl(destination / "source_pool.jsonl", source_pool)
    write_jsonl(destination / "source_support.jsonl", source_subset)
    write_jsonl(destination / "unused_pool_samples.jsonl", unused)
    write_jsonl(destination / "target_test.jsonl", target_test)
    manifest = {
        "source": source,
        "target": target,
        "seed": seed,
        "shot": shot,
        "source_pool": {"normal": 16, "anomalous": 16},
        "source_subset": {"normal": normal_count, "anomalous": anomalous_count},
        "target_test_size": len(target_test),
        "target_used_during_adaptation": False,
    }
    (destination / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="converted MVTec-Periodic directory containing carpet/ and grid/",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", default="0,1,9")
    parser.add_argument("--shots", default="2,4,8,16,32")
    args = parser.parse_args()
    for seed in parse_ints(args.seeds):
        for shot in parse_ints(args.shots):
            if shot not in SHOTS:
                parser.error(f"shots must be selected from {SHOTS}")
            for source, target in DIRECTIONS:
                build_split(args.data_root, args.output_root, source, target, seed, shot)


if __name__ == "__main__":
    main()
