#!/usr/bin/env python3
"""Run one source-only cross-process PG-CLIP experiment.

The four reported configurations are evaluated from matched source/target
metadata: AA-CLIP, Generic Prompts, APSF, and PG-CLIP (APSF + KAN-Residual).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from protocol import assert_target_data_isolation


DIRECTIONS = {
    "active_to_cell": ("AMOLED-Active", "AMOLED-Cell"),
    "cell_to_active": ("AMOLED-Cell", "AMOLED-Active"),
    "carpet_to_grid": ("MVTec-Periodic-carpet", "MVTec-Periodic-grid"),
    "grid_to_carpet": ("MVTec-Periodic-grid", "MVTec-Periodic-carpet"),
}

DATASET_SUBDIRECTORIES = {
    "AMOLED-Active": Path("AMOLED-ARRAY/Active"),
    "AMOLED-Cell": Path("AMOLED-ARRAY/Cell"),
    "MVTec-Periodic-carpet": Path("MVTec-Periodic/carpet"),
    "MVTec-Periodic-grid": Path("MVTec-Periodic/grid"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_shared_text_checkpoint(source: Path, destination: Path) -> str:
    """Copy once and enforce byte-identical Stage-1 text checkpoints."""
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(source, destination)
    source_hash = sha256(source)
    if sha256(destination) != source_hash:
        raise RuntimeError("PG-CLIP must reuse the Generic Prompts text checkpoint")
    return source_hash


def execute(command: list[str], project: Path, env: dict[str, str], dry_run: bool) -> None:
    print("$", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=project, env=env, check=True)


def train_command(
    project: Path,
    dataset: str,
    metadata: Path,
    checkpoint: Path,
    seed: int,
    shot: int,
    prompt_mode: str,
    adapter_type: str,
    apsf_weight: float,
) -> list[str]:
    command = [
        sys.executable,
        str(project / "train.py"),
        "--dataset", dataset,
        "--training_mode", "few_shot",
        "--shot", str(shot),
        "--seed", str(seed),
        "--metadata-path", str(metadata),
        "--save_path", str(checkpoint),
        "--text_batch_size", "64",
        "--image_batch_size", "8",
        "--text_epoch", "5",
        "--image_epoch", "5",
        "--text_lr", "1e-5",
        "--image_lr", "5e-4",
        "--text_adapt_until", "3",
        "--image_adapt_until", "6",
        "--text_adapt_weight", "0.1",
        "--image_adapt_weight", "0.1",
        "--prompt_mode", prompt_mode,
        "--apsf_weight", str(apsf_weight),
        "--image_adapter_type", adapter_type,
        "--image_proj_type", "simple",
    ]
    if adapter_type == "kan_residual":
        command += [
            "--kan_bottleneck", "16",
            "--kan_adapter_bottleneck", "16",
            "--kan_proj_bottleneck", "16",
            "--kan_grid_size", "2",
            "--kan_spline_order", "2",
        ]
    return command


def test_command(
    project: Path,
    dataset: str,
    metadata: Path,
    checkpoint: Path,
    output: Path,
    seed: int,
    shot: int,
    prompt_mode: str,
    adapter_type: str,
    apsf_weight: float,
    visualize: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(project / "test.py"),
        "--dataset", dataset,
        "--shot", str(shot),
        "--seed", str(seed),
        "--metadata-path", str(metadata),
        "--save_path", str(checkpoint),
        "--checkpoint_epoch", "5",
        "--batch_size", "128",
        "--text_adapt_until", "3",
        "--image_adapt_until", "6",
        "--text_adapt_weight", "0.1",
        "--image_adapt_weight", "0.1",
        "--prompt_mode", prompt_mode,
        "--apsf_weight", str(apsf_weight),
        "--image_adapter_type", adapter_type,
        "--image_proj_type", "simple",
        "--metrics_json_path", str(output / "metrics.json"),
        "--log_path", str(output / "test.log"),
    ]
    if adapter_type == "kan_residual":
        command += [
            "--kan_bottleneck", "16",
            "--kan_adapter_bottleneck", "16",
            "--kan_proj_bottleneck", "16",
            "--kan_grid_size", "2",
            "--kan_spline_order", "2",
        ]
    if visualize:
        command += ["--visualize", "--visualize_dir", str(output / "visualizations")]
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="root containing AMOLED-ARRAY/ and/or MVTec-Periodic/",
    )
    parser.add_argument("--direction", choices=sorted(DIRECTIONS), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shot", type=int, choices=(2, 4, 8, 16, 32), required=True)
    parser.add_argument("--split-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project = args.project_root.resolve()
    default_split_name = (
        "amoled_directional_fulltarget_v2"
        if "cell" in args.direction
        else "mvtec_carpet_grid"
    )
    split_root = (args.split_root or project / "splits" / default_split_name).resolve()
    output_root = (args.output_root or project / "results" / "pgclip_protocol").resolve()
    split = split_root / args.direction / f"seed_{args.seed}" / f"shot_{args.shot}"
    source_metadata = split / "source_support.jsonl"
    target_metadata = split / "target_test.jsonl"
    if not args.dry_run:
        for required in (source_metadata, target_metadata, project / "model" / "ViT-L-14-336px.pt"):
            if not required.is_file():
                raise FileNotFoundError(required)

    source_dataset, target_dataset = DIRECTIONS[args.direction]
    if not args.dry_run:
        data_root = args.data_root.resolve()
        assert_target_data_isolation(
            source_metadata,
            target_metadata,
            data_root / DATASET_SUBDIRECTORIES[source_dataset],
            data_root / DATASET_SUBDIRECTORIES[target_dataset],
        )
    run_root = output_root / args.direction / f"seed_{args.seed}" / f"shot_{args.shot}"
    checkpoints = {
        "aa_clip": run_root / "aa_clip" / "checkpoint",
        "generic_prompts": run_root / "generic_prompts" / "checkpoint",
        "pg_clip": run_root / "pg_clip" / "checkpoint",
    }
    if not args.dry_run:
        for path in checkpoints.values():
            path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PGCLIP_DATA_ROOT"] = str(args.data_root.resolve())
    env["PYTHONUNBUFFERED"] = "1"

    training = (
        ("aa_clip", "aa", "simple", 0.0),
        ("generic_prompts", "mvfa", "simple", 0.0),
    )
    for name, prompt, adapter, weight in training:
        final_checkpoint = checkpoints[name] / "image_adapter_5.pth"
        if args.dry_run or not final_checkpoint.is_file():
            execute(
                train_command(
                    project, source_dataset, source_metadata, checkpoints[name],
                    args.seed, args.shot, prompt, adapter, weight,
                ),
                project, env, args.dry_run,
            )

    omega_dir = run_root / "apsf_weight"
    omega_file = omega_dir / "learned_weight.json"
    learn_command = [
        sys.executable,
        str(project / "tools" / "learn_apsf_weight.py"),
        "--source", source_dataset,
        "--shot", str(args.shot),
        "--seed", str(args.seed),
        "--checkpoint-dir", str(checkpoints["generic_prompts"]),
        "--output-dir", str(omega_dir),
        "--metadata-path", str(source_metadata),
        "--epochs", "5",
        "--batch-size", "64",
        "--learning-rate", "0.05",
        "--initial-weight", "0.5",
    ]
    if args.dry_run or not omega_file.is_file():
        execute(learn_command, project, env, args.dry_run)
    omega = 0.5 if args.dry_run else float(json.loads(omega_file.read_text())["final_weight"])

    generic_text = checkpoints["generic_prompts"] / "text_adapter.pth"
    pg_text = checkpoints["pg_clip"] / "text_adapter.pth"
    if not args.dry_run:
        ensure_shared_text_checkpoint(generic_text, pg_text)
    pg_final = checkpoints["pg_clip"] / "image_adapter_5.pth"
    if args.dry_run or not pg_final.is_file():
        execute(
            train_command(
                project, source_dataset, source_metadata, checkpoints["pg_clip"],
                args.seed, args.shot, "mvfa", "kan_residual", 0.0,
            ),
            project, env, args.dry_run,
        )

    evaluations = (
        ("aa_clip", checkpoints["aa_clip"], "aa", "simple", 0.0),
        ("generic_prompts", checkpoints["generic_prompts"], "mvfa", "simple", 0.0),
        ("apsf", checkpoints["generic_prompts"], "apsf", "simple", omega),
        ("pg_clip", checkpoints["pg_clip"], "apsf", "kan_residual", omega),
    )
    for name, checkpoint, prompt, adapter, weight in evaluations:
        output = run_root / name / "target_evaluation"
        if not args.dry_run:
            output.mkdir(parents=True, exist_ok=True)
        if args.dry_run or not (output / "metrics.json").is_file():
            execute(
                test_command(
                    project, target_dataset, target_metadata, checkpoint, output,
                    args.seed, args.shot, prompt, adapter, weight, args.visualize,
                ),
                project, env, args.dry_run,
            )

    if not args.dry_run:
        manifest = {
            "direction": args.direction,
            "seed": args.seed,
            "shot": args.shot,
            "source_dataset": source_dataset,
            "target_dataset": target_dataset,
            "apsf_weight": omega,
            "generic_text_checkpoint_sha256": sha256(generic_text),
            "pg_text_checkpoint_sha256": sha256(pg_text),
            "methods": [item[0] for item in evaluations],
        }
        (run_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(f"completed: {run_root}")


if __name__ == "__main__":
    main()
