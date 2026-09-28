# Adapted from AA-CLIP and modified for PG-CLIP; see NOTICE.
import os
import argparse
import numpy as np
from tqdm import tqdm
import logging
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from utils import setup_seed
from model.adapter import AdaptedCLIP
from model.clip import create_model
from dataset import get_dataset
from forward_utils import (
    get_adapted_text_embedding,
    get_adapted_single_class_text_embedding,
    calculate_similarity_map,
    calculate_seg_loss,
)
import warnings

warnings.filterwarnings("ignore")

cpu_num = 4

os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train_text_adapter(
    adapted_model: nn.Module,
    clip_surgery: nn.Module,
    text_norm_weight: float,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    # scheduler: torch.optim.lr_scheduler,
    device: str,
    start_epoch: int,
    save_path: str,
    text_epoch: int,
    dataset_name: str,
    img_size: int,
    logger: logging.Logger,
):
    # 遍历每一个文本适配器训练 epoch
    for epoch in range(start_epoch, text_epoch):
        logger.info(f"training text epoch {epoch}:")

        loss_list = []
        # 逐个 batch 训练
        for input_data in tqdm(train_loader):
            # 将图像、掩码和类别名从 dataloader 中取出，图像和掩码放到指定设备
            image = input_data["image"].to(device)
            mask = input_data["mask"].to(device)
            class_names = input_data["class_name"]

            # forward text：获取当前 batch 中每个类别的文本嵌入
            epoch_text_feature_dict = {}
            for class_name in list(set(class_names)):
                text_embedding = get_adapted_single_class_text_embedding(
                    adapted_model, dataset_name, class_name, device
                )
                epoch_text_feature_dict[class_name] = text_embedding
            # 按 batch 中样本的类别顺序堆叠文本嵌入，形状为 (bs, 768, 2)
            # 其中 768 是 CLIP ViT-L 文本特征维度，2 表示正常/异常两类文本 anchor
            epoch_text_feature = torch.stack(
                [epoch_text_feature_dict[class_name] for class_name in class_names],
                dim=0,
            )  # bs,768,2

            # forward image：提取图像的多层 patch 特征和全局 cls_token
            with torch.no_grad():
                # clip_surgery 用于提取第 6/12/18/24 层的 patch 特征
                # encode_image 返回 (pooled, patch_tokens):
                #   pooled 形状: (bs, 768)，即全局 cls_token，已完成视觉投影
                #   patch_tokens 中每个 tensor 形状: (bs, 1 + num_patches, 1024)
                #   默认 518x518 图像下 num_patches = 37x37 = 1369
                _, patch_features = clip_surgery.encode_image(image, [6, 12, 18, 24])
                # 原始 CLIP 用于提取全局 cls_token，不返回中间层
                # cls_token 形状: (bs, 768)，已完成视觉投影
                cls_token, _ = adapted_model.clipmodel.encode_image(image, [])
                # 对 cls_token 做 L2 归一化，形状仍为 (bs, 768)
                cls_token = cls_token / cls_token.norm(dim=-1, keepdim=True)
                # 去掉每个 Transformer 输出的 cls 位置，保留 patch 部分
                # t[:, 1:, :] 形状: (bs, num_patches, 1024)
                patch_features = [
                    clip_surgery.visual.ln_post(t[:, 1:, :]) for t in patch_features
                ]
                # 通过视觉投影矩阵将 patch 特征从 1024 维映射到 768 维 joint embedding space
                # 输出形状: (bs, num_patches, 768)
                patch_features = [t @ clip_surgery.visual.proj for t in patch_features]
                # 对投影后的 patch 特征做 L2 归一化，形状仍为 (bs, num_patches, 768)
                patch_features = [
                    t / t.norm(dim=-1, keepdim=True) for t in patch_features
                ]
                # 将全局 cls_token 加到每个 patch 特征上，融合全局上下文
                # cls_token.unsqueeze(1) 形状: (bs, 1, 768) -> 广播到 (bs, num_patches, 768)
                # 相加后 patch_features 形状: (bs, num_patches, 768)
                patch_features = [t + cls_token.unsqueeze(1) for t in patch_features]
            # Stage 1 intentionally supervises only the deepest selected CLIP
            # patch feature (block 24), matching the reported experiments.
            feature = patch_features[-1]
            patch_preds = calculate_similarity_map(feature, epoch_text_feature, img_size)
            loss = calculate_seg_loss(patch_preds, mask)
            orthogonal_loss = (
                (epoch_text_feature[:, :, 0] * epoch_text_feature[:, :, 1])
                .sum(1)
                .mean()
            ) ** 2
            loss += orthogonal_loss * text_norm_weight
            # backward：反向传播并更新文本适配器参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            # scheduler.step()
        # 打印当前 epoch 的平均损失
        logger.info(f"loss: {np.mean(loss_list)}")
        # save checkpoint：保存文本适配器和优化器状态
        ckp_path = os.path.join(save_path, "text_adapter.pth")
        torch.save(
            {
                "epoch": epoch + 1,
                "text_adapter": adapted_model.text_adapter.state_dict(),
                "text_optimizer": optimizer.state_dict(),
            },
            ckp_path,
        )
    # 返回训练后的模型
    return adapted_model


def train_image_adapter(
    model: nn.Module,
    text_embeddings: torch.Tensor,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    device: str,
    start_epoch: int,
    save_path: str,
    image_epoch: int,
    img_size: int,
    logger: logging.Logger,
):
    # 遍历每一个图像适配器训练 epoch
    for epoch in range(start_epoch, image_epoch):
        logger.info(f"training image epoch {epoch}:")
        loss_list = []
        # 逐个 batch 训练
        for input_data in tqdm(train_loader):
            # 将图像、掩码和标签搬到 GPU
            image = input_data["image"].to(device)
            mask = input_data["mask"].to(device)
            label = input_data["label"].to(device)

            # forward text：从 Stage 1 生成的文本 embedding 字典中取出当前 batch 对应的文本特征
            # 形状为 (bs, 768, 2)，2 表示正常/异常两类文本 anchor
            class_names = input_data["class_name"]
            epoch_text_feature = torch.stack(
                [text_embeddings[class_name] for class_name in class_names], dim=0
            )

            # forward image：提取多层 patch 特征和全局检测特征
            # patch_features 是列表，每个元素形状为 (bs, num_patches, 768)
            # det_feature 形状为 (bs, 768)，用于整图异常分类
            patch_features, det_feature = model(image)

            # calculate similarity and get prediction：同时优化分类损失和分割损失
            loss = 0.0
            # 扩展 det_feature 以便与文本特征做点积
            det_feature = det_feature.unsqueeze(1)  # (bs, 1, 768)
            # 计算整图与正常/异常文本的相似度，得到分类预测
            cls_preds = torch.matmul(det_feature, epoch_text_feature)[:, 0]
            # 图像级分类损失
            loss += F.cross_entropy(cls_preds, label)
            # 逐层计算 patch 级别的分割预测和分割损失
            for f in patch_features:
                # text-image alignment：patch 与文本特征对齐，得到分割预测图
                patch_preds = calculate_similarity_map(f, epoch_text_feature, img_size)
                loss += calculate_seg_loss(patch_preds, mask)  # 分割损失
            # backward：反向传播并更新图像适配器参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            # 更新学习率
            scheduler.step()
        logger.info(f"loss: {np.mean(loss_list)}")
        # save checkpoint
        model_dict = {
            "epoch": epoch + 1,
            "image_adapter": model.image_adapter.state_dict(),
            "image_optimizer": optimizer.state_dict(),
            "model_config": model.image_config(),
        }
        torch.save(model_dict, os.path.join(save_path, "image_adapter.pth"))
        if (epoch + 1) % 1 == 0:
            ckp_path = os.path.join(save_path, f"image_adapter_{epoch + 1}.pth")
            torch.save(
                model_dict,
                ckp_path,
            )
    return model


def main():
    parser = argparse.ArgumentParser(description="Training")
    # model
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-L-14-336",
        help="clip model to use (default: ViT-L-14-336)",
    )
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--surgery_until_layer", type=int, default=20)
    parser.add_argument("--relu", action="store_true", help="use relu after projection")
    # training
    parser.add_argument("--dataset", type=str, default="VisA")
    parser.add_argument(
        "--training_mode",
        type=str,
        default="few_shot",
        choices=["few_shot", "full_shot"],
    )
    parser.add_argument("--shot", type=int, default=32, help="number of shots (0 means full shot)")
    parser.add_argument(
        "--metadata-path",
        type=str,
        default=None,
        help="job-local JSONL metadata; overrides the shared dataset metadata",
    )
    parser.add_argument("--text_batch_size", type=int, default=16)
    parser.add_argument("--image_batch_size", type=int, default=2)
    parser.add_argument("--text_epoch", type=int, default=5, help="epochs for stage1")
    parser.add_argument("--image_epoch", type=int, default=5, help="epochs for stage2")
    parser.add_argument("--text_lr", type=float, default=0.00001, help="learning rate for stage1")
    parser.add_argument("--image_lr", type=float, default=0.0005, help="learning rate for stage2")
    parser.add_argument(
        "--criterion", type=str, default=["dice_loss", "focal_loss"], nargs="+"
    )
    # exp
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--save_path", type=str, default="ckpt/baseline")
    # hyper-parameters
    parser.add_argument("--text_norm_weight", type=float, default=0.1)
    parser.add_argument("--text_adapt_weight", type=float, default=0.1)
    parser.add_argument("--image_adapt_weight", type=float, default=0.1)
    parser.add_argument("--text_adapt_until", type=int, default=3)
    parser.add_argument("--image_adapt_until", type=int, default=6)
    parser.add_argument(
        "--adapter_type",
        choices=["simple", "kan_residual"],
        default="simple",
        help="fallback type for image adapter and projection",
    )
    parser.add_argument(
        "--image_adapter_type",
        choices=["simple", "kan_residual"],
        default=None,
        help="override the visual residual adapter type",
    )
    parser.add_argument(
        "--image_proj_type",
        choices=["simple", "kan_residual"],
        default=None,
        help="override the segmentation and detection projection type",
    )
    parser.add_argument("--kan_bottleneck", type=int, default=16)
    parser.add_argument("--kan_adapter_bottleneck", type=int, default=None)
    parser.add_argument("--kan_proj_bottleneck", type=int, default=None)
    parser.add_argument("--kan_grid_size", type=int, default=3)
    parser.add_argument("--kan_spline_order", type=int, default=3)
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default=os.environ.get("PROMPT_MODE", "mvfa"),
        choices=[
            "aa",
            "mvfa",
            "apsf",
        ],
    )
    parser.add_argument(
        "--apsf_weight",
        type=float,
        default=float(os.environ.get("APSF_WEIGHT", "0.5")),
    )

    args = parser.parse_args()
    if not 0.0 <= args.apsf_weight <= 1.0:
        parser.error("--apsf_weight must be in [0, 1]")
    os.environ["PROMPT_MODE"] = args.prompt_mode
    os.environ["APSF_WEIGHT"] = str(args.apsf_weight)
    args.image_adapter_type = args.image_adapter_type or args.adapter_type
    args.image_proj_type = args.image_proj_type or args.adapter_type
    args.kan_adapter_bottleneck = args.kan_adapter_bottleneck or args.kan_bottleneck
    args.kan_proj_bottleneck = args.kan_proj_bottleneck or args.kan_bottleneck
    if any(
        value <= 0
        for value in (
            args.kan_bottleneck,
            args.kan_adapter_bottleneck,
            args.kan_proj_bottleneck,
        )
    ):
        parser.error("KAN bottlenecks must be positive")
    if args.kan_grid_size <= 0 or args.kan_spline_order <= 0:
        parser.error("KAN grid size and spline order must be positive")
    # ========================================================
    setup_seed(args.seed)
    # check save_path and setting logger
    os.makedirs(args.save_path, exist_ok=True)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(args.save_path, "train.log"),
        encoding="utf-8",
        level=logging.INFO,
    )
    logger.info("args: %s", vars(args))
    # set device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # ========================================================
    # load model：加载 CLIP 模型并构建适配器模型
    # 创建用于提取多层 patch 特征的 CLIP Surgery 模型
    clip_surgery = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_surgery.eval()
    # 对 CLIP Surgery 视觉编码器进行 DAPM 替换，用于提取中间层 patch 特征
    clip_surgery.visual.DAPM_replace(DPAM_layer=args.surgery_until_layer)
    # 创建用于训练的基础 CLIP 模型
    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_model.eval()
    # 构建 AdaptedCLIP，封装 text_adapter 和 image_adapter
    model = AdaptedCLIP(
        clip_model=clip_model,
        text_adapt_weight=args.text_adapt_weight,
        image_adapt_weight=args.image_adapt_weight,
        text_adapt_until=args.text_adapt_until,
        image_adapt_until=args.image_adapt_until,
        relu=args.relu,
        image_adapter_type=args.image_adapter_type,
        image_proj_type=args.image_proj_type,
        kan_bottleneck=args.kan_bottleneck,
        kan_adapter_bottleneck=args.kan_adapter_bottleneck,
        kan_proj_bottleneck=args.kan_proj_bottleneck,
        kan_grid_size=args.kan_grid_size,
        kan_spline_order=args.kan_spline_order,
        kan_seed=args.seed,
    ).to(device)
    model.eval()
    # 设置优化器：分别优化文本适配器和图像适配器
    text_optimizer = torch.optim.Adam(
        model.text_trainable_parameters(),
        lr=args.text_lr,
        betas=(0.5, 0.999),
    )
    image_optimizer = torch.optim.Adam(
        model.image_adapter.parameters(),
        lr=args.image_lr,
        betas=(0.5, 0.999),
    )
    # text_scheduler = MultiStepLR(text_optimizer, milestones=[400], gamma=0.1)
    # 图像适配器学习率调度器
    image_scheduler = MultiStepLR(image_optimizer, milestones=[16000, 32000], gamma=0.5)
    # ========================================================
    # load checkpoints if exists：断点续训，恢复 text/image adapter 和优化器状态
    text_file = glob(args.save_path + "/text_adapter.pth")
    if len(text_file) > 0:
        checkpoint = torch.load(text_file[0])
        model.text_adapter.load_state_dict(checkpoint["text_adapter"])
        try:
            text_optimizer.load_state_dict(checkpoint["text_optimizer"])
        except Exception as exc:
            logger.warning("skip text optimizer state restore: %s", exc)
        text_start_epoch = checkpoint["epoch"]
        # 如果已经训练完所有 text_epoch，则跳过文本适配器训练
        adapt_text = not (text_start_epoch == (args.text_epoch - 1))
    elif args.text_epoch == 0:
        adapt_text = False
    else:
        text_start_epoch = 0
        adapt_text = True  # 需要从头训练文本适配器
    # 恢复图像适配器 checkpoint
    file = glob(args.save_path + "/image_adapter.pth")
    if len(file) > 0:
        checkpoint = torch.load(file[0])
        image_start_epoch = checkpoint["epoch"]
        model.image_adapter.load_state_dict(checkpoint["image_adapter"])
        image_optimizer.load_state_dict(checkpoint["image_optimizer"])
    else:
        image_start_epoch = 0
    # ========================================================
    # load dataset：加载训练数据集
    if args.training_mode == "full_shot":
        args.shot = -1
    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}
    logger.info("loading dataset ...")
    # 返回两个数据集：text_dataset 用于训练文本适配器，image_dataset 用于训练图像适配器
    text_dataset, image_dataset = get_dataset(
        args.dataset,
        args.img_size,
        args.training_mode,
        args.shot,
        "train",
        logger,
        metadata_path=args.metadata_path,
    )
    source_class_names = sorted(
        {record["class_name"] for record in image_dataset.meta}
    )
    text_dataloader = torch.utils.data.DataLoader(
        text_dataset, batch_size=args.text_batch_size, shuffle=True, **kwargs
    )
    logger.info("loading image adaptation dataset ...")
    image_dataloader = torch.utils.data.DataLoader(
        image_dataset, batch_size=args.image_batch_size, shuffle=True, **kwargs
    )
    # ========================================================
    # training：执行两阶段训练
    # Stage 1：训练文本适配器
    if adapt_text:
        model = train_text_adapter(
            adapted_model=model,
            clip_surgery=clip_surgery,
            text_norm_weight=args.text_norm_weight,
            train_loader=text_dataloader,
            optimizer=text_optimizer,
            # scheduler=text_scheduler,
            device=device,
            start_epoch=text_start_epoch,
            dataset_name=args.dataset,
            save_path=args.save_path,
            text_epoch=args.text_epoch,
            img_size=args.img_size,
            logger=logger,
        )
    # 释放文本阶段占用的内存
    del text_dataloader, text_dataset, clip_surgery, text_optimizer
    torch.cuda.empty_cache()
    # 用训练好的文本适配器生成每个类别的文本 embedding，供 Stage 2 使用
    with torch.no_grad():
        if args.text_epoch == 0:
            # 不训练文本适配器时，直接用原始 CLIP 生成文本 embedding
            text_embeddings = get_adapted_text_embedding(
                clip_model, args.dataset, device, source_class_names
            )
        else:
            # 用训练好的 AdaptedCLIP 生成文本 embedding
            text_embeddings = get_adapted_text_embedding(
                model, args.dataset, device, source_class_names
            )
    # Stage 2：训练图像适配器
    model = train_image_adapter(
        model=model,
        text_embeddings=text_embeddings,
        image_epoch=args.image_epoch,
        train_loader=image_dataloader,
        optimizer=image_optimizer,
        scheduler=image_scheduler,
        device=device,
        start_epoch=image_start_epoch,
        save_path=args.save_path,
        img_size=args.img_size,
        logger=logger,
    )


if __name__ == "__main__":
    main()
