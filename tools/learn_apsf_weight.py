#!/usr/bin/env python3
"""Learn only the APSF MVFA-to-PA fusion weight on a source-domain split."""

import argparse
import hashlib
import json
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import get_dataset
from forward_utils import (
    calculate_seg_loss,
    calculate_similarity_map,
    get_adapted_single_class_text_embedding,
)
from model.adapter import AdaptedCLIP
from model.clip import create_model
from utils import setup_seed


def logit(value: float) -> float:
    return math.log(value / (1.0 - value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--shot", type=int, required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--img-size", type=int, default=518)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--initial-weight", type=float, default=0.5)
    parser.add_argument("--text-norm-weight", type=float, default=0.1)
    parser.add_argument("--text-adapt-weight", type=float, default=0.1)
    parser.add_argument("--image-adapt-weight", type=float, default=0.1)
    parser.add_argument("--text-adapt-until", type=int, default=3)
    parser.add_argument("--image-adapt-until", type=int, default=6)
    parser.add_argument("--surgery-until-layer", type=int, default=20)
    parser.add_argument("--model-name", default="ViT-L-14-336")
    args = parser.parse_args()

    if args.shot <= 0:
        parser.error("--shot must be positive")
    if not 0.0 < args.initial_weight < 1.0:
        parser.error("--initial-weight must be strictly between 0 and 1")
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        parser.error("epochs, batch size, and learning rate must be positive")

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    text_checkpoint_path = checkpoint_dir / "text_adapter.pth"
    if not text_checkpoint_path.is_file():
        raise FileNotFoundError(f"missing baseline checkpoint: {text_checkpoint_path}")

    os.environ["PROMPT_MODE"] = "apsf"
    setup_seed(args.seed)
    logging.basicConfig(
        filename=output_dir / "learn_weight.log",
        encoding="utf-8",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    logger.info("args: %s", vars(args))

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    clip_surgery = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_surgery.eval()
    clip_surgery.visual.DAPM_replace(DPAM_layer=args.surgery_until_layer)

    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_model.eval()
    model = AdaptedCLIP(
        clip_model=clip_model,
        text_adapt_weight=args.text_adapt_weight,
        image_adapt_weight=args.image_adapt_weight,
        text_adapt_until=args.text_adapt_until,
        image_adapt_until=args.image_adapt_until,
        relu=False,
    ).to(device)
    checkpoint = torch.load(text_checkpoint_path, map_location=device)
    model.text_adapter.load_state_dict(checkpoint["text_adapter"])
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in clip_surgery.parameters():
        parameter.requires_grad_(False)

    apsf_logit = torch.nn.Parameter(
        torch.tensor(logit(args.initial_weight), dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(
        [apsf_logit], lr=args.learning_rate, betas=(0.5, 0.999)
    )
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if optimizer_parameter_ids != {id(apsf_logit)}:
        raise RuntimeError("optimizer must contain only apsf_logit")

    text_dataset, _ = get_dataset(
        args.source,
        args.img_size,
        "few_shot",
        args.shot,
        "train",
        logger,
        metadata_path=args.metadata_path,
    )
    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        text_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs,
    )

    trace = []
    for epoch in range(args.epochs):
        loss_values = []
        for input_data in train_loader:
            image = input_data["image"].to(device)
            mask = input_data["mask"].to(device)
            class_names = input_data["class_name"]
            apsf_weight = torch.sigmoid(apsf_logit)

            epoch_text_feature_dict = {}
            for class_name in set(class_names):
                epoch_text_feature_dict[class_name] = (
                    get_adapted_single_class_text_embedding(
                        model,
                        args.source,
                        class_name,
                        device,
                        apsf_weight=apsf_weight,
                    )
                )
            epoch_text_feature = torch.stack(
                [epoch_text_feature_dict[name] for name in class_names], dim=0
            )

            with torch.no_grad():
                _, patch_features = clip_surgery.encode_image(
                    image, [6, 12, 18, 24]
                )
                cls_token, _ = model.clipmodel.encode_image(image, [])
                cls_token = cls_token / cls_token.norm(dim=-1, keepdim=True)
                patch_features = [
                    clip_surgery.visual.ln_post(feature[:, 1:, :])
                    for feature in patch_features
                ]
                patch_features = [
                    feature @ clip_surgery.visual.proj for feature in patch_features
                ]
                patch_features = [
                    feature / feature.norm(dim=-1, keepdim=True)
                    for feature in patch_features
                ]
                patch_features = [
                    feature + cls_token.unsqueeze(1) for feature in patch_features
                ]

            # Match Stage 1: supervise the deepest selected feature only.
            feature = patch_features[-1]
            patch_preds = calculate_similarity_map(
                feature, epoch_text_feature, args.img_size
            )
            loss = calculate_seg_loss(patch_preds, mask)
            orthogonal_loss = (
                (epoch_text_feature[:, :, 0] * epoch_text_feature[:, :, 1])
                .sum(1)
                .mean()
            ) ** 2
            loss += orthogonal_loss * args.text_norm_weight

            optimizer.zero_grad()
            loss.backward()
            if apsf_logit.grad is None:
                raise RuntimeError("apsf_logit did not receive a gradient")
            if any(parameter.grad is not None for parameter in model.parameters()):
                raise RuntimeError("a frozen model parameter received a gradient")
            optimizer.step()
            loss_values.append(float(loss.detach().cpu()))

        epoch_row = {
            "epoch": epoch + 1,
            "loss": float(np.mean(loss_values)),
            "apsf_logit": float(apsf_logit.detach().cpu()),
            "apsf_weight": float(torch.sigmoid(apsf_logit).detach().cpu()),
        }
        trace.append(epoch_row)
        logger.info("epoch: %s", epoch_row)
        torch.save(
            {
                "epoch": epoch + 1,
                "apsf_logit": apsf_logit.detach().cpu(),
                "optimizer": optimizer.state_dict(),
            },
            output_dir / "apsf_weight.pth",
        )

    payload = {
        "source": args.source,
        "shot": args.shot,
        "seed": args.seed,
        "metadata_path": str(Path(args.metadata_path).resolve()),
        "metadata_sha256": sha256_file(Path(args.metadata_path)),
        "initial_weight": args.initial_weight,
        "final_weight": trace[-1]["apsf_weight"],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "optimizer": "Adam",
        "betas": [0.5, 0.999],
        "loss": "deepest-feature Focal+Dice plus anchor orthogonality",
        "trainable_parameters": ["apsf_logit"],
        "trace": trace,
    }
    (output_dir / "learned_weight.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
