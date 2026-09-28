# Adapted from AA-CLIP and modified for PG-CLIP; see NOTICE.
import numpy as np
import cv2
import os
import hashlib
from pathlib import Path
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from kornia.filters import gaussian_blur2d
from dataset.constants import (
    CLASS_NAMES,
    REAL_NAMES,
    PROMPTS,
    MVFA_PROMPTS,
    PERIODICITY_AWARE_PROMPTS,
)
from model.tokenizer import tokenize
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
import pandas as pd
from dataset.constants import DATA_PATH

# ================================================================================================
# The following code is used to get criterion for training


class FocalLoss(nn.Module):
    """Multiclass focal loss for probability maps with label smoothing."""

    def __init__(self, gamma: float = 2.0, smooth: float = 1e-5):
        super().__init__()
        self.gamma = gamma
        self.smooth = smooth
        if not 0.0 <= smooth <= 1.0:
            raise ValueError("smooth must be in [0, 1]")

    def forward(self, probabilities: torch.Tensor, target: torch.Tensor):
        classes = probabilities.shape[1]
        if classes < 2:
            raise ValueError("focal loss requires at least two classes")
        if probabilities.ndim > 2:
            probabilities = probabilities.movedim(1, -1).reshape(-1, classes)
        labels = target.squeeze(1).reshape(-1).long()
        targets = F.one_hot(labels, num_classes=classes).to(probabilities)
        if self.smooth:
            targets = targets.clamp(
                self.smooth / (classes - 1), 1.0 - self.smooth
            )
        pt = (targets * probabilities).sum(dim=1) + self.smooth
        return (-(1.0 - pt).pow(self.gamma) * pt.log()).mean()


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        N = targets.size()[0]
        smooth = 1
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)
        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (
            input_flat.sum(1) + targets_flat.sum(1) + smooth
        )
        loss = 1 - N_dice_eff.sum() / N
        return loss


# ================================================================================================
# The following code is used to get adapted text embeddings

def _get_prompt_mode():
    mode = os.environ.get("PROMPT_MODE", "mvfa").lower()
    valid_modes = {"aa", "mvfa", "apsf"}
    if mode not in valid_modes:
        raise ValueError(
            f"Unknown PROMPT_MODE={mode}; available modes: {sorted(valid_modes)}"
        )
    return mode


def _get_prompt_config(dataset_name):
    mode = _get_prompt_mode()
    return PROMPTS if mode == "aa" else MVFA_PROMPTS


def _expand_prompt_config(prompt, real_name, state_idx):
    prompt_state = [prompt["prompt_normal"], prompt["prompt_abnormal"]]
    prompt_templates = prompt["prompt_templates"]
    prompted_state = [state.format(real_name) for state in prompt_state[state_idx]]
    prompted_sentence = []
    for state in prompted_state:
        for template in prompt_templates:
            prompted_sentence.append(template.format(state))
    return prompted_sentence


def _build_prompted_sentences(dataset_name, real_name, state_idx):
    return _expand_prompt_config(_get_prompt_config(dataset_name), real_name, state_idx)


def _encode_prompt_anchor(model, prompted_sentence, device):
    prompted_sentence = tokenize(prompted_sentence).to(device)
    class_embeddings = model.encode_text(prompted_sentence)
    class_embeddings = class_embeddings / class_embeddings.norm(
        dim=-1, keepdim=True
    )
    class_embedding = class_embeddings.mean(dim=0)
    return class_embedding / class_embedding.norm()


def _encode_state_anchor(
    model,
    dataset_name,
    real_name,
    state_idx,
    device,
    apsf_weight=None,
):
    mode = _get_prompt_mode()
    if mode == "apsf":
        mvfa_embedding = _encode_prompt_anchor(
            model, _expand_prompt_config(MVFA_PROMPTS, real_name, state_idx), device
        )
        # The normal anchor remains exactly MVFA-like. The PA branch refines
        # only the abnormal concept, where generic words such as "damaged"
        # and "defect" are least informative.
        if state_idx == 0:
            return mvfa_embedding
        pa_weight = apsf_weight
        if pa_weight is None:
            pa_weight = float(os.environ.get("APSF_WEIGHT", "0.05"))
        weight_value = (
            float(pa_weight.detach().cpu().item())
            if torch.is_tensor(pa_weight)
            else float(pa_weight)
        )
        if not 0.0 <= weight_value <= 1.0:
            raise ValueError(
                f"APSF_WEIGHT must be in [0, 1], got {weight_value}"
            )
        pa_embedding = _encode_prompt_anchor(
            model,
            _expand_prompt_config(PERIODICITY_AWARE_PROMPTS, real_name, state_idx),
            device,
        )
        class_embedding = (
            (1.0 - pa_weight) * mvfa_embedding
            + pa_weight * pa_embedding
        )
        return class_embedding / class_embedding.norm()
    return _encode_prompt_anchor(
        model, _build_prompted_sentences(dataset_name, real_name, state_idx), device
    )


def get_adapted_single_class_text_embedding(
    model,
    dataset_name,
    class_name,
    device,
    apsf_weight=None,
):
    if class_name == "object":
        real_name = class_name
    else:
        real_name = REAL_NAMES[dataset_name].get(
            class_name, REAL_NAMES[dataset_name][CLASS_NAMES[dataset_name][0]]
        )
    text_features = []
    for i in range(2):
        text_features.append(
            _encode_state_anchor(
                model,
                dataset_name,
                real_name,
                i,
                device,
                apsf_weight=apsf_weight,
            )
        )
    text_features = torch.stack(text_features, dim=1).to(device)
    return text_features


def get_adapted_single_sentence_text_embedding(model, dataset_name, class_name, device):
    real_name = REAL_NAMES[dataset_name].get(
        class_name, REAL_NAMES[dataset_name][CLASS_NAMES[dataset_name][0]]
    )
    text_features = []
    for i in range(2):
        prompted_sentence = _build_prompted_sentences(dataset_name, real_name, i)
        prompted_sentence = tokenize(prompted_sentence).to(device)
        class_embeddings = model.encode_text(prompted_sentence)
        class_embeddings = F.normalize(class_embeddings, dim=-1)
        text_features.append(class_embeddings)
    text_features = torch.cat(text_features, dim=0).to(device)
    return text_features


def get_adapted_text_embedding(model, dataset_name, device, class_names=None):
    ret_dict = {}
    for class_name in class_names or CLASS_NAMES[dataset_name]:
        text_features = get_adapted_single_class_text_embedding(
            model, dataset_name, class_name, device
        )
        ret_dict[class_name] = text_features
    return ret_dict


# ================================================================================================
def calculate_similarity_map(
    patch_features, epoch_text_feature, img_size, test=False, domain="Medical"
):
    patch_anomaly_scores = 100.0 * torch.matmul(patch_features, epoch_text_feature)
    B, L, C = patch_anomaly_scores.shape
    H = int(np.sqrt(L))
    patch_pred = patch_anomaly_scores.permute(0, 2, 1).view(B, C, H, H)
    if test:
        assert C == 2
        sigma = 1 if domain == "Industrial" else 1.5
        kernel_size = 7 if domain == "Industrial" else 9
        patch_pred = (patch_pred[:, 1] + 1 - patch_pred[:, 0]) / 2
        patch_pred = gaussian_blur2d(
            patch_pred.unsqueeze(1), (kernel_size, kernel_size), (sigma, sigma)
        )
    patch_preds = F.interpolate(
        patch_pred, size=img_size, mode="bilinear", align_corners=True
    )
    if not test and C > 1:
        patch_preds = torch.softmax(patch_preds, dim=1)
    return patch_preds


def abnormal_anchor_image_score(
    image_features: torch.Tensor, text_anchors: torch.Tensor
) -> torch.Tensor:
    """Return the historical image score based on the abnormal anchor."""
    similarities = image_features @ text_anchors
    return (similarities[:, 1] + 1.0) / 2.0


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    """Min--max normalise scores while handling constant arrays safely."""
    values = np.asarray(values)
    minimum = values.min()
    span = values.max() - minimum
    return np.zeros_like(values, dtype=np.float64) if span == 0 else (values - minimum) / span


def combine_industrial_image_scores(
    pixel_predictions: np.ndarray, image_predictions: np.ndarray
) -> np.ndarray:
    """Fuse the maximum pixel score and semantic image score equally."""
    pixel_maximum = pixel_predictions.max(axis=(1, 2))
    return 0.5 * pixel_maximum + 0.5 * image_predictions


focal_loss = FocalLoss()
dice_loss = BinaryDiceLoss()


def calculate_seg_loss(patch_preds, mask):
    loss = focal_loss(patch_preds, mask)
    loss += dice_loss(patch_preds[:, 0, :, :], 1 - mask)
    loss += dice_loss(patch_preds[:, 1, :, :], mask)
    return loss


# ================================================================================================


def metrics_eval(
    pixel_label: np.ndarray,
    image_label: np.ndarray,
    pixel_preds: np.ndarray,
    image_preds: np.ndarray,
    class_names: str,
    domain: str,
):
    if pixel_preds.max() != 1:
        pixel_preds = minmax_normalize(pixel_preds)
    if image_preds.max() != 1:
        image_preds = minmax_normalize(image_preds)

    pmax_pred = pixel_preds.max(axis=(1, 2))
    if domain != "Medical":
        image_preds = combine_industrial_image_scores(pixel_preds, image_preds)
    else:
        image_preds = pmax_pred
    # ================================================================================================
    # pixel level auc & ap
    pixel_label = pixel_label.flatten()
    pixel_preds = pixel_preds.flatten()

    zero_pixel_auc = roc_auc_score(pixel_label, pixel_preds)
    zero_pixel_ap = average_precision_score(pixel_label, pixel_preds)
    # ================================================================================================
    # image level auc & ap
    if image_label.max() != image_label.min():
        image_label = image_label.flatten()
        agg_image_preds = image_preds.flatten()
        agg_image_auc = roc_auc_score(image_label, agg_image_preds)
        agg_image_ap = average_precision_score(image_label, agg_image_preds)
    else:
        agg_image_auc = 0
        agg_image_ap = 0
    def f1_max(labels, scores):
        precision, recall, _ = precision_recall_curve(labels, scores)
        denominator = precision + recall
        f1 = np.divide(
            2 * precision * recall,
            denominator,
            out=np.zeros_like(precision, dtype=np.float64),
            where=denominator > 0,
        )
        return float(f1.max()) if f1.size else 0.0

    pixel_f1_max = f1_max(pixel_label, pixel_preds)
    image_f1_max = f1_max(image_label, image_preds) if image_label.max() != image_label.min() else 0.0
    # ================================================================================================
    result = {
        "class name": class_names,
        "pixel AUC": round(zero_pixel_auc, 4) * 100,
        "pixel AP": round(zero_pixel_ap, 4) * 100,
        "pixel F1 max": round(pixel_f1_max, 4) * 100,
        "image AUC": round(agg_image_auc, 4) * 100,
        "image AP": round(agg_image_ap, 4) * 100,
        "image F1 max": round(image_f1_max, 4) * 100,
    }
    return result


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    return (alpha * image + (1 - alpha) * scoremap).astype(np.uint8)


def visualize(
    pixel_label: np.ndarray,
    pixel_preds: np.ndarray,
    file_names: list[str],
    save_dir: str,
    dataset_name: str,
    class_name: str,
):
    pixel_preds = np.asarray(pixel_preds)
    finite_preds = np.nan_to_num(
        pixel_preds.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    pred_min = float(finite_preds.min())
    pred_max = float(finite_preds.max())
    if pred_max > pred_min:
        pixel_preds = (
            (finite_preds - pred_min) / (pred_max - pred_min) * 255
        ).astype(np.uint8)
    else:
        # Constant maps are valid, but must not produce NaNs during scaling.
        pixel_preds = np.zeros_like(finite_preds, dtype=np.uint8)

    pixel_label = np.asarray(pixel_label)
    if pixel_label.dtype != np.uint8 or pixel_label.max(initial=0) <= 1:
        pixel_label = ((pixel_label != 0) * 255).astype(np.uint8)
    # ===============================================================================================
    # save path
    save_dir = os.path.join(save_dir, "visualization", dataset_name, class_name)
    os.makedirs(save_dir, exist_ok=True)
    written_files = []
    for idx, file in enumerate(file_names):
        image_file = os.path.join(DATA_PATH[dataset_name], file)
        image = cv2.imread(image_file)
        if image is None:
            raise FileNotFoundError(f"unable to read visualization image: {image_file}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (pixel_label.shape[-1], pixel_label.shape[-2]))
        save_image_list = [image]

        if dataset_name == "MVTec":
            damage_name, image_name = file.split("/")[-2:]
            file_name = f"{damage_name}_{image_name}"
        else:
            # AMOLED metadata contains nested relative paths.  Hash the full
            # metadata path so duplicate basenames cannot overwrite one another.
            relative_path = Path(str(file).replace("\\", "/"))
            digest = hashlib.sha1(str(file).encode("utf-8")).hexdigest()[:12]
            file_name = f"{digest}_{relative_path.name}"

        save_image_list.append(cv2.cvtColor(pixel_label[idx, 0], cv2.COLOR_GRAY2RGB))
        save_image_list.append(cv2.cvtColor(pixel_preds[idx], cv2.COLOR_GRAY2RGB))
        save_image_list = save_image_list[:1] + [
            apply_ad_scoremap(image, _) for _ in save_image_list[1:]
        ]
        scoremap = np.vstack(save_image_list)
        output_path = os.path.join(save_dir, file_name)
        if not cv2.imwrite(output_path, scoremap):
            raise IOError(f"unable to write visualization image: {output_path}")
        written_files.append(output_path)
    return written_files
