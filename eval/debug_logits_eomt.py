import os
import sys
sys.path.insert(0, "/content/MaskArchitectureAnomaly_CourseProject/eomt")

import warnings
import importlib
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from torch.amp.autocast_mode import autocast
import yaml

# ── ripreso da evalAnomaly.py ──────────────────────────────────────────────────

def load_model(config_path, device, img_size=None, num_classes=None):
    from huggingface_hub import hf_hub_download

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_kwargs = config["data"].get("init_args", {})
    if img_size is None:
        img_size = data_kwargs.get("img_size", (640, 640))
    img_size = tuple(img_size)
    if num_classes is None:
        num_classes = data_kwargs.get("num_classes", 19)

    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    enc_mod, enc_cls_name = encoder_cfg["class_path"].rsplit(".", 1)
    encoder_cls = getattr(importlib.import_module(enc_mod), enc_cls_name)
    encoder = encoder_cls(img_size=img_size, **encoder_cfg.get("init_args", {}))

    network_cfg = config["model"]["init_args"]["network"]
    net_mod, net_cls_name = network_cfg["class_path"].rsplit(".", 1)
    network_cls = getattr(importlib.import_module(net_mod), net_cls_name)
    network_kwargs = {k: v for k, v in network_cfg["init_args"].items() if k != "encoder"}
    network = network_cls(
        masked_attn_enabled=False,
        num_classes=num_classes,
        encoder=encoder,
        **network_kwargs,
    )

    lit_mod, lit_cls_name = config["model"]["class_path"].rsplit(".", 1)
    lit_cls = getattr(importlib.import_module(lit_mod), lit_cls_name)
    model_kwargs = {k: v for k, v in config["model"]["init_args"].items() if k != "network"}

    if "stuff_classes" in data_kwargs:
        model_kwargs["stuff_classes"] = data_kwargs["stuff_classes"]

    warnings.filterwarnings("ignore",
        message=r".*Attribute 'network' is an instance of `nn\.Module`.*")

    model = lit_cls(
        img_size=img_size,
        num_classes=num_classes,
        network=network,
        **model_kwargs,
    ).eval().to(device)

    ckpt_path = "/content/drive/MyDrive/eomt_weights/eomt_cityscapes.bin"
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    print("Weights loaded.")
    return model, img_size


def compute_scores(model, img_tensor, img_size, device):
    """
    Ritorna un dict con tutte le anomaly maps: MSP, Entropy, MaxLogit, Energy, RbA.
    """
    dtype = torch.float16 if device != "cpu" else torch.float32

    with torch.no_grad(), autocast(dtype=dtype, device_type=device):
        imgs      = [img_tensor.to(device)]
        img_sizes = [img_tensor.shape[-2:]]
        crops, origins = model.window_imgs_semantic(imgs)
        mask_logits_per_layer, class_logits_per_layer = model(crops)

        mask_logits = F.interpolate(
            mask_logits_per_layer[-1], img_size,
            mode="bilinear", align_corners=False
        )

        # pixel logits via pipeline ufficiale
        crop_logits  = model.to_per_pixel_logits_semantic(
            mask_logits, class_logits_per_layer[-1]
        )
        logits       = model.revert_window_logits_semantic(crop_logits, origins, img_sizes)
        pixel_logits = logits[0].float()  # [C, H, W]

        # per RbA servono anche mask_logits rivertiti
        mask_logits_rev = model.revert_window_logits_semantic(
            mask_logits, origins, img_sizes
        )[0].float()                                # [Q, H, W]
        class_logits_q  = class_logits_per_layer[-1][0].float()  # [Q, C+1]

    # ── baseline pixel-level ──────────────────────────────────────────────────
    probs = torch.softmax(pixel_logits, dim=0)          # [C, H, W]

    msp     = (1.0 - probs.max(dim=0)[0]).cpu().numpy()
    entropy = (-torch.sum(probs * torch.log(probs + 1e-8), dim=0)).cpu().numpy()
    maxlogit= (-pixel_logits.max(dim=0)[0]).cpu().numpy()
    T = 1.0
    energy  = (-T * torch.logsumexp(pixel_logits / T, dim=0)).cpu().numpy()

    # ── RbA (formula da paper: tanh-based) ───────────────────────────────────
    sigma    = (torch.tanh(pixel_logits) + 1.0) / 2.0   # [C, H, W]
    rba      = (-sigma.sum(dim=0)).cpu().numpy()          # [H, W]

    return {
        "MSP":      msp,
        "Entropy":  entropy,
        "MaxLogit": maxlogit,
        "Energy":   energy,
        "RbA":      rba,
    }


# ── visualizzazione ────────────────────────────────────────────────────────────

BASELINES = ["MSP", "Entropy", "MaxLogit", "Energy", "RbA"]
CMAPS     = ["hot", "inferno", "plasma", "magma", "viridis"]


def visualize_image(pil_img, scores, title, gt_mask=None):
    """
    Riga superiore: immagine originale (+ GT opzionale).
    Riga inferiore: una heatmap per ogni baseline.
    """
    n_baselines = len(BASELINES)
    n_cols      = max(n_baselines, 2)   # almeno 2 colonne per la riga top
    n_rows      = 2

    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                           hspace=0.35, wspace=0.25)

    # ── riga 0: immagine + GT ─────────────────────────────────────────────────
    ax_img = fig.add_subplot(gs[0, :2])
    ax_img.imshow(pil_img)
    ax_img.set_title("Input image")
    ax_img.axis("off")

    if gt_mask is not None:
        ax_gt = fig.add_subplot(gs[0, 2:4] if n_cols >= 4 else gs[0, 2:])
        im_gt = ax_gt.imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
        ax_gt.set_title("GT (white = anomaly)")
        ax_gt.axis("off")
        plt.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)

    # ── riga 1: heatmaps ──────────────────────────────────────────────────────
    for i, (name, cmap) in enumerate(zip(BASELINES, CMAPS)):
        ax = fig.add_subplot(gs[1, i])
        score = scores[name]
        # normalizza in [0,1] per confronto visivo omogeneo
        s_min, s_max = score.min(), score.max()
        score_norm   = (score - s_min) / (s_max - s_min + 1e-8)
        im = ax.imshow(score_norm, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(name, fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.show()


def overlay_heatmap(pil_img, score, title, alpha=0.55, cmap="hot"):
    """Overlay della heatmap sull'immagine originale (utile per un confronto rapido)."""
    s_min, s_max = score.min(), score.max()
    score_norm   = (score - s_min) / (s_max - s_min + 1e-8)

    cmap_fn   = plt.get_cmap(cmap)
    heatmap   = cmap_fn(score_norm)[..., :3]           # [H, W, 3] float
    img_arr   = np.array(pil_img).astype(np.float32) / 255.0
    blended   = (1 - alpha) * img_arr + alpha * heatmap

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.imshow(np.clip(blended, 0, 1))
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── configurazione ────────────────────────────────────────────────────────
    CONFIG   = "../configs/dinov2/cityscapes/semantic/eomt_base_640.yaml"
    IMG_SIZE = (640, 640)
    DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

    # immagini da visualizzare: (path_immagine, path_gt_opzionale, etichetta)
    IMAGES = [
        (
            "/content/drive/MyDrive/Anomaly_Segmentation_Datasets/"
            "Validation_Dataset/RoadAnomaly21/images/8.png",
            "/content/drive/MyDrive/Anomaly_Segmentation_Datasets/"
            "Validation_Dataset/RoadAnomaly21/labels_masks/8.png",
            "RoadAnomaly21 – img 8",
        ),
        (
            "/content/drive/MyDrive/Anomaly_Segmentation_Datasets/"
            "Validation_Dataset/fs_static/images/2.jpg",
            None,
            "fs_static – img 2 (normale)",
        ),
    ]

    # ── carica modello ────────────────────────────────────────────────────────
    print(f"Device: {DEVICE}")
    model, img_size = load_model(CONFIG, DEVICE, img_size=IMG_SIZE)

    input_resize = lambda pil: pil.resize((img_size[1], img_size[0]), Image.BILINEAR)
    gt_resize    = lambda pil: pil.resize((img_size[1], img_size[0]), Image.NEAREST)

    # ── inferenza e plot ──────────────────────────────────────────────────────
    for img_path, gt_path, label in IMAGES:
        print(f"\n{'─'*60}")
        print(f"Image: {label}")

        pil_img    = input_resize(Image.open(img_path).convert("RGB"))
        img_tensor = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1)  # uint8 [C,H,W]

        gt_mask = None
        if gt_path and os.path.exists(gt_path):
            gt_raw  = np.array(gt_resize(Image.open(gt_path).convert("L")))
            # remap RoadAnomaly21 (label 2 → 1)
            if "RoadAnomaly" in gt_path:
                gt_raw = np.where(gt_raw == 2, 1, gt_raw)
            gt_mask = (gt_raw == 1).astype(np.uint8)

        scores = compute_scores(model, img_tensor, img_size, DEVICE)

        # 1. griglia con tutte le heatmap
        visualize_image(pil_img, scores, title=label, gt_mask=gt_mask)

        # 2. overlay MSP e RbA sull'immagine (le due più informative visivamente)
        for name in ("MSP", "RbA"):
            overlay_heatmap(pil_img, scores[name],
                            title=f"Overlay {name} – {label}")

        torch.cuda.empty_cache()
