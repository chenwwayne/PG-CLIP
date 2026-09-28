"""Protocol invariants shared by runners and tests."""

from __future__ import annotations

import json
from pathlib import Path


def metadata_paths(metadata: Path, data_root: Path) -> set[Path]:
    paths = set()
    with metadata.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                paths.add((data_root / record["image_path"]).resolve())
    return paths


def assert_target_data_isolation(
    source_metadata: Path,
    target_metadata: Path,
    source_root: Path,
    target_root: Path,
) -> None:
    """Reject source/target image overlap before any adaptation command runs."""
    overlap = metadata_paths(source_metadata, source_root) & metadata_paths(
        target_metadata, target_root
    )
    if overlap:
        sample = sorted(map(str, overlap))[:3]
        raise RuntimeError(f"source/target metadata overlap: {sample}")
