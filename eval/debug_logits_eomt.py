# scritp per printare heatmap di anomaly detection

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

# medesima struttura di evalAnomaly

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

        # per RbA servono anche mask_logits reverted
        mask_logits_rev = model.revert_window_logits_semantic(
            mask_logits, origins, img_sizes )[0].float()                                # [Q, H, W]
        class_logits_q  = class_logits_per_layer[-1][0].float()  # [Q, C+1]

    # baselines
    probs = torch.softmax(pixel_logits, dim=0)          # [C, H, W]

    msp     = (1.0 - probs.max(dim=0)[0]).cpu().numpy()
    entropy = (-torch.sum(probs * torch.log(probs + 1e-8), dim=0)).cpu().numpy()
    maxlogit= (-pixel_logits.max(dim=0)[0]).cpu().numpy()
    T = 1.0
    energy  = (-T * torch.logsumexp(pixel_logits / T, dim=0)).cpu().numpy()

    # RbA
    sigma    = (torch.tanh(pixel_logits) + 1.0) / 2.0   # [C, H, W]
    rba      = (-sigma.sum(dim=0)).cpu().numpy()          # [H, W]

    return {
        "MSP":      msp,
        "Entropy":  entropy,
        "MaxLogit": maxlogit,
        "Energy":   energy,
        "RbA":      rba,
    }, pixel_logits.cpu(), class_logits_q.cpu(), mask_logits_rev.cpu()


# visualizzazione immagini

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

    # riga 0: immagine + GT 
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

    # riga 1: heatmaps 
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

    safe_title = title.replace(" – ", "_").replace(" ", "_").replace("–", "_")
    nome_file = f"griglia_{safe_title}.png"
    
    plt.savefig(nome_file, bbox_inches='tight', dpi=150)
    print(f"Grafico griglia salvato in: {nome_file}")
    
    plt.close()


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
    
    safe_title = title.replace(" – ", "_").replace(" ", "_").replace("–", "_")
    nome_file = f"{safe_title}.png"
    
    plt.savefig(nome_file, bbox_inches='tight', dpi=150)
    print(f"Overlay salvato in: {nome_file}")
    plt.close()


def plot_ts_comparison(pil_img, scores_base, scores_qts, gt_mask=None, title=""):
    """
    Confronto visivo baseline vs query-TS.
    scores_base e scores_qts sono dict con chiave 'MSP'.
    """
    fig, axes = plt.subplots(1, 4 if gt_mask is not None else 3, figsize=(18, 4))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    axes[0].imshow(pil_img)
    axes[0].set_title("Input image")
    axes[0].axis("off")

    col = 1
    if gt_mask is not None:
        axes[col].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
        axes[col].set_title("GT (white = anomaly)")
        axes[col].axis("off")
        col += 1

    for score, label in [
        (scores_base["MSP"], "MSP  T=1.0\n(baseline)"),
        (scores_qts["MSP"],  "MSP  query-T=4.0"),
    ]:
        s = (score - score.min()) / (score.max() - score.min() + 1e-8)
        im = axes[col].imshow(s, cmap="hot", vmin=0, vmax=1)
        axes[col].set_title(label)
        axes[col].axis("off")
        plt.colorbar(im, ax=axes[col], fraction=0.046, pad=0.04)
        col += 1

    plt.tight_layout()
    safe_title = title.replace(" – ", "_").replace(" ", "_").replace("–", "_")
    plt.savefig(f"ts_comparison_{safe_title}.png", bbox_inches="tight", dpi=150)
    print(f"Saved: ts_comparison_{safe_title}.png")
    plt.show()


def compute_scores_query_ts(pixel_logits, class_logits, mask_probs, t):
    """
    Applica query-level temperature scaling a partire dai logit cachati.
    pixel_logits: [C, H, W], class_logits: [Q, C+1], mask_probs: [Q, H, W]
    """
    class_probs_t    = torch.softmax(class_logits / t, dim=-1)   # [Q, C+1]
    class_probs_id_t = class_probs_t[:, :-1]                      # [Q, C]
    pixel_probs_t    = torch.einsum("qc,qhw->chw",
                                    class_probs_id_t, mask_probs) # [C, H, W]

    msp = (1.0 - pixel_probs_t.max(dim=0)[0]).numpy()

    pixel_probs_norm = pixel_probs_t / (pixel_probs_t.sum(0, keepdim=True) + 1e-8)
    entropy = (-torch.sum(
        pixel_probs_norm * torch.log(pixel_probs_norm + 1e-8), dim=0
    )).numpy()

    return {"MSP": msp, "Entropy": entropy}


def plot_rba_failure(pil_img, scores, gt_mask=None, title="RbA failure case"):
    """
    Mostra affiancati MSP (funziona) e RbA (fallisce) per evidenziare il problema.
    """
    fig, axes = plt.subplots(1, 4 if gt_mask is not None else 3, figsize=(18, 4))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    axes[0].imshow(pil_img)
    axes[0].set_title("Input image")
    axes[0].axis("off")

    col = 1
    if gt_mask is not None:
        axes[col].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
        axes[col].set_title("GT (white = anomaly)")
        axes[col].axis("off")
        col += 1

    for name, cmap, note in [
        ("MSP", "hot",     "FPR@95 ≈ 0.55%  ✓"),
        ("RbA", "viridis", "FPR@95 ≈ 99.95%  ✗"),
    ]:
        s = scores[name]
        s_norm = (s - s.min()) / (s.max() - s.min() + 1e-8)
        im = axes[col].imshow(s_norm, cmap=cmap, vmin=0, vmax=1)
        axes[col].set_title(f"{name}\n{note}", fontsize=10)
        axes[col].axis("off")
        plt.colorbar(im, ax=axes[col], fraction=0.046, pad=0.04)
        col += 1

    plt.tight_layout()
    safe_title = title.replace(" – ", "_").replace(" ", "_").replace("–", "_")
    plt.savefig(f"rba_failure_{safe_title}.png", bbox_inches="tight", dpi=150)
    print(f"Saved: rba_failure_{safe_title}.png")
    plt.show()


def plot_query_ts_cross_dataset(
    pil_road21, scores_road21_base, scores_road21_qts,
    pil_obstacle, scores_obstacle_base, scores_obstacle_qts,
):
    """
    2 righe x 3 colonne:
      riga 0 → RoadAnomaly21  (query-TS funziona)
      riga 1 → RoadObstacle21 (query-TS fallisce)
    colonne: immagine | MSP T=1.0 | MSP query-T=4.0
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Query-TS: dataset dove funziona vs dove fallisce",
                 fontsize=13, fontweight="bold")

    rows = [
        (pil_road21,   scores_road21_base,   scores_road21_qts,
         "RoadAnomaly21", "AUPRC 84.35%  FPR 3.63%  ✓"),
        (pil_obstacle, scores_obstacle_base, scores_obstacle_qts,
         "RoadObstacle21", "AUPRC 63.45%  FPR 100%  ✗"),
    ]

    for row_idx, (pil, base, qts, dataset_name, note) in enumerate(rows):
        axes[row_idx, 0].imshow(pil)
        axes[row_idx, 0].set_title(dataset_name, fontsize=10)
        axes[row_idx, 0].axis("off")

        for col_idx, (score, label) in enumerate([
            (base["MSP"], "MSP  T=1.0"),
            (qts["MSP"],  f"MSP  query-T=4.0\n{note}"),
        ]):
            s = (score - score.min()) / (score.max() - score.min() + 1e-8)
            im = axes[row_idx, col_idx + 1].imshow(s, cmap="hot", vmin=0, vmax=1)
            axes[row_idx, col_idx + 1].set_title(label, fontsize=9)
            axes[row_idx, col_idx + 1].axis("off")
            plt.colorbar(im, ax=axes[row_idx, col_idx + 1],
                         fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("query_ts_cross_dataset.png", bbox_inches="tight", dpi=150)
    print("Saved: query_ts_cross_dataset.png")
    plt.show()


def plot_prediction_comparison(pil_img, pixel_logits, scores, gt_mask=None, title=""):
    """
    4 pannelli: immagine originale | segmentation prediction | GT mask | anomaly map (MSP)
    pixel_logits: [C, H, W] tensor CPU
    """
    # palette Cityscapes (19 classi)
    CITYSCAPES_COLORS = np.array([
        [128, 64,128], [244, 35,232], [ 70, 70, 70], [102,102,156],
        [190,153,153], [153,153,153], [250,170, 30], [220,220,  0],
        [107,142, 35], [152,251,152], [ 70,130,180], [220, 20, 60],
        [255,  0,  0], [  0,  0,142], [  0,  0, 70], [  0, 60,100],
        [  0, 80,100], [  0,  0,230], [119, 11, 32]
    ], dtype=np.uint8)

    pred_class = pixel_logits.argmax(dim=0).numpy()          # [H, W]
    pred_color = CITYSCAPES_COLORS[pred_class % 19]          # [H, W, 3]

    n_cols = 4 if gt_mask is not None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # col 0 — immagine originale
    axes[0].imshow(pil_img)
    axes[0].set_title("Input image")
    axes[0].axis("off")

    # col 1 — segmentation prediction
    axes[1].imshow(pred_color)
    axes[1].set_title("Semantic prediction")
    axes[1].axis("off")

    col = 2
    # col 2 — GT mask (opzionale)
    if gt_mask is not None:
        axes[col].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
        axes[col].set_title("GT anomaly mask")
        axes[col].axis("off")
        col += 1

    # col 3 — anomaly map MSP
    msp = scores["MSP"]
    msp_norm = (msp - msp.min()) / (msp.max() - msp.min() + 1e-8)
    im = axes[col].imshow(msp_norm, cmap="hot", vmin=0, vmax=1)
    axes[col].set_title("Anomaly map (MSP)")
    axes[col].axis("off")
    plt.colorbar(im, ax=axes[col], fraction=0.046, pad=0.04)

    plt.tight_layout()
    safe_title = title.replace(" – ", "_").replace(" ", "_").replace("–", "_")
    plt.savefig(f"pred_comparison_{safe_title}.png", bbox_inches="tight", dpi=150)
    print(f"Saved: pred_comparison_{safe_title}.png")
    plt.close()



# MAIN

if __name__ == "__main__":
    # configurazione 
    CONFIG   = "../eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml"
    IMG_SIZE = (1024,1024)
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
            "Validation_Dataset/RoadObsticle21/images/5.webp",
            None,
            "RoadObsticle21 – img 5 ",
        ),
    ]

    # carica modello 
    print(f"Device: {DEVICE}")
    model, img_size = load_model(CONFIG, DEVICE, img_size=IMG_SIZE)

    input_resize = lambda pil: pil.resize((img_size[1], img_size[0]), Image.BILINEAR)
    gt_resize    = lambda pil: pil.resize((img_size[1], img_size[0]), Image.NEAREST)

   

    # inferenza e plot 
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

        scores, pixel_logits, class_logits, mask_probs = compute_scores(model, img_tensor, img_size, DEVICE)

        # 1. griglia con tutte le heatmap
        visualize_image(pil_img, scores, title=label, gt_mask=gt_mask)

        plot_prediction_comparison(pil_img, pixel_logits, scores, gt_mask=gt_mask, title=label)

        # 2. overlay MSP e RbA sull'immagine (le due più informative visivamente)
        for name in ("MSP", "RbA"):
            overlay_heatmap(pil_img, scores[name],
                            title=f"Overlay {name} – {label}")

        scores_qts = compute_scores_query_ts(pixel_logits, class_logits, mask_probs, t=4.0)

# figura 1 — comparativa TS (solo su RoadAnomaly21)
        if "RoadAnomaly21" in img_path:
            pil_road= pil_img
            scores_road= scores
            scores_road_q= scores_qts
            plot_ts_comparison(pil_img, scores, scores_qts, gt_mask=gt_mask,
                                title=f"Baseline vs Query-TS — {label}")

# figura 2 — fallimento RbA (su RoadObstacle21 o fs_static)
        if "RoadObsticle21" in img_path or "fs_static" in img_path:
            pil_obs= pil_img
            scores_obs= scores
            scores_obs_q= scores_qts
            plot_rba_failure(pil_img, scores, gt_mask=gt_mask, title=f"RbA failure — {label}")

        torch.cuda.empty_cache()
   
plot_query_ts_cross_dataset(pil_road, scores_road, scores_road_q, pil_obs, scores_obs, scores_obs_q)
