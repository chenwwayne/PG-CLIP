#!/usr/bin/env python3
"""Create carpet/grid domain proxies from an official MVTec AD tree."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


CATEGORIES = ("carpet", "grid")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def transfer(source: Path, destination: Path, mode: str) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(source.resolve())


def images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def prepare_category(source_root: Path, output_root: Path, category: str, mode: str) -> dict:
    source = source_root / category
    output = output_root / category
    normal_count = 0
    anomaly_count = 0

    for split in ("train", "test"):
        for image in images(source / split / "good"):
            name = f"{split}_good_{image.name}"
            transfer(image, output / "images" / "normal" / name, mode)
            normal_count += 1

    test_root = source / "test"
    for defect_dir in sorted(path for path in test_root.iterdir() if path.is_dir()):
        if defect_dir.name == "good":
            continue
        ground_truth = source / "ground_truth" / defect_dir.name
        for image in images(defect_dir):
            name = f"{defect_dir.name}_{image.name}"
            mask = ground_truth / f"{image.stem}_mask.png"
            if not mask.is_file():
                raise FileNotFoundError(f"missing mask for {image}: {mask}")
            transfer(image, output / "images" / "anomaly" / name, mode)
            transfer(mask, output / "mask" / "anomaly" / name, mode)
            anomaly_count += 1

    return {"normal": normal_count, "anomalous": anomaly_count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mvtec-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("copy", "symlink"), default="copy")
    args = parser.parse_args()

    expected = {"carpet": (308, 89), "grid": (285, 57)}
    for category in CATEGORIES:
        counts = prepare_category(args.mvtec_root, args.output_root, category, args.mode)
        actual = (counts["normal"], counts["anomalous"])
        if actual != expected[category]:
            raise RuntimeError(
                f"{category} count mismatch: expected {expected[category]}, got {actual}"
            )
        print(f"{category}: {counts}")


if __name__ == "__main__":
    main()
