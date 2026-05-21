# Anomaly evaluation script adapted for EoMT
# Based on the original ERFNet evalAnomaly.py

import os
import glob
import torch
import random
import warnings
import importlib
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os.path as osp
from argparse import ArgumentParser
from torch.cuda.amp import autocast
from ood_metrics import fpr_at_95_tpr
from sklearn.metrics import average_precision_score

# ── reproducibility ──────────────────────────────────────────────────────────
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

NUM_CLASSES = 19  # Cityscapes classes (EoMT uses 19, not 20 like ERFNet)
IGNORE_INDEX = 255


# ── helpers ───────────────────────────────────────────────────────────────────

def load_model(config_path: str, weights_path: str, device: str):
    """
    Instantiate and load an EoMT Lightning model from a YAML config + checkpoint.
    Mirrors the loading logic in the official EoMT notebook.
    """
    import yaml
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import RepositoryNotFoundError

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # ── resolve img_size and num_classes from the config ────────────────────
    data_cfg = config.get("data", {}).get("init_args", {})
    img_size = data_cfg.get("img_size", (512, 1024))
    num_classes = data_cfg.get("num_classes", NUM_CLASSES)

    # ── build encoder ────────────────────────────────────────────────────────
    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    enc_mod, enc_cls_name = encoder_cfg["class_path"].rsplit(".", 1)
    encoder_cls = getattr(importlib.import_module(enc_mod), enc_cls_name)
    encoder = encoder_cls(img_size=img_size, **encoder_cfg.get("init_args", {}))

    # ── build network ────────────────────────────────────────────────────────
    network_cfg = config["model"]["init_args"]["network"]
    net_mod, net_cls_name = network_cfg["class_path"].rsplit(".", 1)
    network_cls = getattr(importlib.import_module(net_mod), net_cls_name)
    network_kwargs = {k: v for k, v in network_cfg["init_args"].items() if k != "encoder"}
    network = network_cls(
        masked_attn_enabled=False,   # disabled at inference as per EoMT docs
        num_classes=num_classes,
        encoder=encoder,
        **network_kwargs,
    )

    # ── build Lightning module ───────────────────────────────────────────────
    lit_mod, lit_cls_name = config["model"]["class_path"].rsplit(".", 1)
    lit_cls = getattr(importlib.import_module(lit_mod), lit_cls_name)
    model_kwargs = {k: v for k, v in config["model"]["init_args"].items() if k != "network"}
    if "stuff_classes" in data_cfg:
        model_kwargs["stuff_classes"] = data_cfg["stuff_classes"]

    model = lit_cls(
        img_size=img_size,
        num_classes=num_classes,
        network=network,
        **model_kwargs,
    ).eval().to(device)

    # ── load weights ─────────────────────────────────────────────────────────
    if weights_path and os.path.isfile(weights_path):
        print(f"Loading weights from local file: {weights_path}")
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    else:
        # fallback: try HuggingFace Hub using the logger name in config
        name = (config.get("trainer", {})
                      .get("logger", {})
                      .get("init_args", {})
                      .get("name"))
        if name:
            print(f"Trying to download weights from HuggingFace Hub: tue-mps/{name}")
            try:
                hf_path = hf_hub_download(repo_id=f"tue-mps/{name}",
                                           filename="pytorch_model.bin")
                state_dict = torch.load(hf_path, map_location=device, weights_only=True)
                model.load_state_dict(state_dict, strict=False)
                print("Weights loaded from HuggingFace Hub.")
            except RepositoryNotFoundError:
                warnings.warn(f"No pre-trained model found for '{name}'. "
                              "Proceeding with random weights.")
        else:
            warnings.warn("No weights path provided and no HuggingFace name in config. "
                          "Proceeding with random weights.")

    return model, img_size


def infer_single(model, img_tensor: torch.Tensor, img_size, device: str):
    """
    Run EoMT semantic inference on a single image tensor (C,H,W).
    Returns:
        anomaly_msp    (H, W) float32 – 1 - max softmax prob
        anomaly_logit  (H, W) float32 – -max logit
        anomaly_entropy(H, W) float32 – predictive entropy
    """
    with torch.no_grad(), autocast(dtype=torch.float16, device_type="cuda"):
        imgs = [img_tensor.to(device)]
        img_sizes = [img_tensor.shape[-2:]]

        # EoMT sliding-window preprocessing
        crops, origins = model.window_imgs_semantic(imgs)

        # Forward pass – returns lists (one entry per decoder layer)
        mask_logits_per_layer, class_logits_per_layer = model(crops)

        # Use only the last layer's predictions (best quality)
        mask_logits = F.interpolate(
            mask_logits_per_layer[-1], img_size, mode="bilinear", align_corners=False
        )
        class_logits = class_logits_per_layer[-1]   # (B, Q, num_classes+1)

        # Per-pixel logits via the official EoMT combiner
        # shape: (B, num_classes, H, W)
        pixel_logits = model.to_per_pixel_logits_semantic(mask_logits, class_logits)

        # Revert window tiling back to original image size
        pixel_logits = model.revert_window_logits_semantic(
            pixel_logits, origins, img_sizes
        )                                            # list of (num_classes, H, W)
        pixel_logits = pixel_logits[0]               # (num_classes, H, W)

    # Cast back to float32 for metric computation
    pixel_logits = pixel_logits.float()

    # ── MSP anomaly score ────────────────────────────────────────────────────
    probs = torch.softmax(pixel_logits, dim=0)          # (C, H, W)
    anomaly_msp = (1.0 - torch.max(probs, dim=0)[0])    # (H, W)

    # ── MaxLogit anomaly score ───────────────────────────────────────────────
    anomaly_logit = -torch.max(pixel_logits, dim=0)[0]  # (H, W)

    # ── Entropy anomaly score ────────────────────────────────────────────────
    anomaly_entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=0)  # (H, W)

    return (
        anomaly_msp.cpu().numpy(),
        anomaly_logit.cpu().numpy(),
        anomaly_entropy.cpu().numpy(),
    )


def remap_gt(ood_gts: np.ndarray, pathGT: str) -> np.ndarray:
    """Apply dataset-specific label remapping to ground-truth masks."""
    if "RoadAnomaly" in pathGT:
        ood_gts = np.where(ood_gts == 2, 1, ood_gts)
    if "LostAndFound" in pathGT:
        ood_gts = np.where(ood_gts == 0, 255, ood_gts)
        ood_gts = np.where(ood_gts == 1, 0, ood_gts)
        ood_gts = np.where((ood_gts > 1) & (ood_gts < 201), 1, ood_gts)
    if "Streethazard" in pathGT:
        ood_gts = np.where(ood_gts == 14, 255, ood_gts)
        ood_gts = np.where(ood_gts < 20, 0, ood_gts)
        ood_gts = np.where(ood_gts == 255, 1, ood_gts)
    return ood_gts


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/content/datasets/RoadAnomaly21/images/*.png",
        nargs="+",
        help="Glob pattern for input images, e.g. 'path/to/images/*.png'",
    )
    parser.add_argument("--config",  default="../configs/dinov2/cityscapes/semantic/eomt_large_640.yaml",
                        help="Path to the EoMT YAML config file")
    parser.add_argument("--weights", default="../trained_models/pytorch_model.bin",
                        help="Path to pytorch_model.bin checkpoint (or leave empty to use HuggingFace Hub)")
    parser.add_argument("--img-size", type=int, nargs=2, default=None,
                        help="Override inference image size as H W, e.g. --img-size 512 1024")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = "cpu" if args.cpu else "cuda"

    # ── load model ────────────────────────────────────────────────────────────
    print(f"Loading EoMT from config: {args.config}")
    model, img_size = load_model(args.config, args.weights, device)
    if args.img_size is not None:
        img_size = tuple(args.img_size)
    print(f"Model loaded. Inference image size: {img_size}")
    print("Model and weights LOADED successfully")

    # ── results file ──────────────────────────────────────────────────────────
    if not os.path.exists("results.txt"):
        open("results.txt", "w").close()
    result_file = open("results.txt", "a")

    # ── image preprocessing (no external normalisation: EoMT does it internally) ──
    from torchvision.transforms import Compose, Resize, ToTensor
    input_transform = Compose([
        Resize(img_size, Image.BILINEAR),
        ToTensor(),
        # NOTE: EoMT normalises internally via pixel_mean / pixel_std in forward()
    ])
    target_transform = Compose([
        Resize(img_size, Image.NEAREST),
    ])

    # ── per-image loop ────────────────────────────────────────────────────────
    anomaly_score_list         = []
    anomaly_score_list_logit   = []
    anomaly_score_list_entropy = []
    ood_gts_list               = []

    pattern = os.path.expanduser(str(args.input[0]))
    paths   = sorted(glob.glob(pattern))

    if not paths:
        parent = os.path.dirname(pattern)
        print(f"No images found with pattern: {pattern}")
        if os.path.exists(parent):
            print(f"Contents of {parent}: {os.listdir(parent)}")
        else:
            print(f"Directory does not exist: {parent}")
        result_file.close()
        return

    print(f"Found {len(paths)} images.")

    for path in paths:
        print(f"Processing: {path}")

        img_tensor = input_transform(Image.open(path).convert("RGB"))  # (C, H, W)

        anomaly_msp, anomaly_logit, anomaly_entropy = infer_single(
            model, img_tensor, img_size, device
        )

        # ── ground-truth mask path ────────────────────────────────────────────
        pathGT = path.replace("images", "labels_masks")
        if "RoadObsticle21" in pathGT:
            pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
            pathGT = pathGT.replace("jpg", "png")
        if "RoadAnomaly" in pathGT:
            pathGT = pathGT.replace("jpg", "png")

        mask    = Image.open(pathGT).convert("L")   # force grayscale
        mask    = target_transform(mask)
        ood_gts = np.array(mask)
        ood_gts = remap_gt(ood_gts, pathGT)

        print(f"  GT path: {pathGT}")
        print(f"  Unique GT values: {np.unique(ood_gts)}")

        # skip images with no OOD pixels
        if 1 not in np.unique(ood_gts):
            print("  No OOD pixels – skipping.")
            continue

        ood_gts_list.append(ood_gts)
        anomaly_score_list.append(anomaly_msp)
        anomaly_score_list_logit.append(anomaly_logit)
        anomaly_score_list_entropy.append(anomaly_entropy)

        torch.cuda.empty_cache()

    if len(ood_gts_list) == 0:
        print("No valid images processed. Check paths and GT masks.")
        result_file.close()
        return

    # ── aggregate & evaluate ──────────────────────────────────────────────────
    ood_gts            = np.array(ood_gts_list)             # (N, H, W)
    anomaly_scores     = np.array(anomaly_score_list)        # (N, H, W)
    anomaly_scores_logit   = np.array(anomaly_score_list_logit)
    anomaly_scores_entropy = np.array(anomaly_score_list_entropy)

    # valid pixels only (exclude ignore index 255)
    valid_mask = (ood_gts != IGNORE_INDEX)
    ood_mask   = (ood_gts == 1) & valid_mask
    ind_mask   = (ood_gts == 0) & valid_mask

    def compute_metrics(scores, ood_m, ind_m, label):
        ood_out   = scores[ood_m]
        ind_out   = scores[ind_m]
        val_out   = np.concatenate([ind_out, ood_out])
        val_label = np.concatenate([np.zeros(len(ind_out)), np.ones(len(ood_out))])
        auprc = average_precision_score(val_label, val_out)
        fpr   = fpr_at_95_tpr(val_out, val_label)
        print(f"[{label}]  AUPRC: {auprc*100:.2f}%   FPR@TPR95: {fpr*100:.2f}%")
        return auprc, fpr

    result_file.write("\n")

    auprc_msp,     fpr_msp     = compute_metrics(anomaly_scores,         ood_mask, ind_mask, "MSP")
    auprc_logit,   fpr_logit   = compute_metrics(anomaly_scores_logit,   ood_mask, ind_mask, "MaxLogit")
    auprc_entropy, fpr_entropy = compute_metrics(anomaly_scores_entropy, ood_mask, ind_mask, "Entropy")

    result_file.write(
        f"  AUPRC (MSP): {auprc_msp*100:.2f}%   FPR@TPR95 (MSP): {fpr_msp*100:.2f}%"
        f"  |  AUPRC (MaxLogit): {auprc_logit*100:.2f}%   FPR@TPR95 (MaxLogit): {fpr_logit*100:.2f}%"
        f"  |  AUPRC (Entropy): {auprc_entropy*100:.2f}%   FPR@TPR95 (Entropy): {fpr_entropy*100:.2f}%\n"
    )
    result_file.close()


if __name__ == "__main__":
    main()
