
# Anomaly evaluation script adapted for EoMT
# Based on the original ERFNet evalAnomaly.py

import os
import sys

# Add EoMT repo root to path so that 'datasets', 'models', 'training' modules are importable
sys.path.insert(0, "/content/MaskArchitectureAnomaly_CourseProject/eomt")

import glob
import torch
import random
import warnings
import importlib
import torch.nn.functional as F
from PIL import Image
import numpy as np
from argparse import ArgumentParser
from torch.amp.autocast_mode import autocast
from ood_metrics import fpr_at_95_tpr
from sklearn.metrics import average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor

# ── reproducibility ───────────────────────────────────────────────────────────
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

IGNORE_INDEX = 255


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(config_path: str, device: str, img_size=None, num_classes=None):
    """
    Instantiate and load an EoMT Lightning model from a YAML config.
    - img_size and num_classes can be passed explicitly (no Cityscapes zip needed)
    - weights are downloaded automatically from HuggingFace Hub using the logger name in the config
    """
    import yaml
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import RepositoryNotFoundError

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_kwargs = config["data"].get("init_args", {})

    # img_size and num_classes: use explicit values if provided, otherwise read from config
    # For cityscapes_semantic eomt_base_640: img_size=(640,640), num_classes=19
    if img_size is None:
        img_size = data_kwargs.get("img_size", (640, 640))
    if num_classes is None:
        num_classes = data_kwargs.get("num_classes", 19)

    print(f"  img_size={img_size}  num_classes={num_classes}")

    # ── build encoder ─────────────────────────────────────────────────────────
    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    enc_mod, enc_cls_name = encoder_cfg["class_path"].rsplit(".", 1)
    encoder_cls = getattr(importlib.import_module(enc_mod), enc_cls_name)
    encoder = encoder_cls(img_size=img_size, **encoder_cfg.get("init_args", {}))

    # ── build network ─────────────────────────────────────────────────────────
    network_cfg = config["model"]["init_args"]["network"]
    net_mod, net_cls_name = network_cfg["class_path"].rsplit(".", 1)
    network_cls = getattr(importlib.import_module(net_mod), net_cls_name)
    network_kwargs = {k: v for k, v in network_cfg["init_args"].items() if k != "encoder"}
    network = network_cls(
        masked_attn_enabled=False,  # always disabled at inference (EoMT docs)
        num_classes=num_classes,
        encoder=encoder,
        **network_kwargs,
    )

    # ── build Lightning module ────────────────────────────────────────────────
    lit_mod, lit_cls_name = config["model"]["class_path"].rsplit(".", 1)
    lit_cls = getattr(importlib.import_module(lit_mod), lit_cls_name)
    model_kwargs = {k: v for k, v in config["model"]["init_args"].items() if k != "network"}

    # pass stuff_classes if present in data config (needed for panoptic)
    if "stuff_classes" in data_kwargs:
        model_kwargs["stuff_classes"] = data_kwargs["stuff_classes"]

    warnings.filterwarnings(
        "ignore",
        message=r".*Attribute 'network' is an instance of `nn\.Module` and is already saved during checkpointing.*",
    )

    model = lit_cls(
        img_size=img_size,
        num_classes=num_classes,
        network=network,
        **model_kwargs,
    ).eval().to(device)

    # ── download weights from HuggingFace Hub ─────────────────────────────────
    name = (config.get("trainer", {})
                  .get("logger", {})
                  .get("init_args", {})
                  .get("name"))

    if name is None:
        warnings.warn("No logger name found in config — proceeding with random weights.")
        return model, img_size

    print(f"Downloading weights from HuggingFace Hub: tue-mps/{name}")
    try:
        is_dinov3 = "dinov3" in name

        # Try safetensors first (newer HuggingFace format), fallback to pytorch_model.bin
        try:
            from safetensors.torch import load_file as safetensors_load
            state_dict_path = hf_hub_download(
                repo_id=f"tue-mps/{name}",
                filename="model.safetensors",
            )
            use_safetensors = True
        except Exception:
            state_dict_path = hf_hub_download(
                repo_id=f"tue-mps/{name}",
                filename="pytorch_model.bin",
            )
            use_safetensors = False

        if is_dinov3:
            # dinov3 models use delta weights — rebuild model with ckpt_path
            model_kwargs["ckpt_path"]      = state_dict_path
            model_kwargs["delta_weights"]  = True
            model = lit_cls(
                img_size=img_size,
                num_classes=num_classes,
                network=network,
                **model_kwargs,
            ).eval().to(device)
        else:
            if use_safetensors:
                state_dict = safetensors_load(state_dict_path, device=device)
            else:
                state_dict = torch.load(state_dict_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict, strict=False)

        print("Weights loaded successfully from HuggingFace Hub.")

    except RepositoryNotFoundError:
        warnings.warn(
            f"Pre-trained model not found for '{name}' on HuggingFace Hub. "
            "Proceeding with random weights."
        )

    return model, img_size


# ── single-image inference ────────────────────────────────────────────────────

def infer_single(model, img_tensor: torch.Tensor, img_size, device: str, temperature=1.0 ):
    """
    Run EoMT semantic inference on a single image tensor (C, H, W).
    NOTE: no external normalisation — EoMT applies pixel_mean/pixel_std internally.

    Returns three (H, W) float32 numpy arrays:
        anomaly_msp     – 1 - max softmax probability  (MSP)
        anomaly_logit   – negative max logit            (MaxLogit)
        anomaly_entropy – predictive entropy            (Entropy)
    """
    dtype = torch.float16 if device != "cpu" else torch.float32

    with torch.no_grad(), autocast(dtype=dtype, device_type=device):
        imgs      = [img_tensor.to(device)]
        img_sizes = [img_tensor.shape[-2:]]

        # sliding-window preprocessing (official EoMT pipeline)
        crops, origins = model.window_imgs_semantic(imgs)

        # forward — returns one entry per decoder layer
        mask_logits_per_layer, class_logits_per_layer = model(crops)

        # use last layer (best quality)
        mask_logits  = F.interpolate(
            mask_logits_per_layer[-1], img_size, mode="bilinear", align_corners=False
        )
        class_logits = class_logits_per_layer[-1]   # (B, Q, num_classes+1)

        # combine masks and class logits → (B, num_classes, H, W)
        pixel_logits = model.to_per_pixel_logits_semantic(mask_logits, class_logits)

        # revert window tiling → list[(num_classes, H, W)]
        pixel_logits = model.revert_window_logits_semantic(
            pixel_logits, origins, img_sizes
        )[0]                                        # (num_classes, H, W)

    pixel_logits = pixel_logits.float() / temperature           # back to fp32 for metrics

    # MSP
    probs           = torch.softmax(pixel_logits, dim=0)
    anomaly_msp     = (1.0 - torch.max(probs, dim=0)[0]).cpu().numpy()

    # MaxLogit
    anomaly_logit   = (-torch.max(pixel_logits, dim=0)[0]).cpu().numpy()

    # Entropy
    anomaly_entropy = (-torch.sum(probs * torch.log(probs + 1e-8), dim=0)).cpu().numpy()

    return anomaly_msp, anomaly_logit, anomaly_entropy

# funzione per calcolare rba 
def infer_single_rba(model, img_tensor, img_size, device):
    dtype = torch.float16 if device != "cpu" else torch.float32

    with torch.no_grad(), autocast(dtype=dtype, device_type=device):
        imgs      = [img_tensor.to(device)]
        img_sizes = [img_tensor.shape[-2:]]
        crops, origins = model.window_imgs_semantic(imgs)
        mask_logits_per_layer, class_logits_per_layer = model(crops)

        mask_logits  = F.interpolate(
            mask_logits_per_layer[-1], img_size, mode="bilinear", align_corners=False
        )
        class_logits = class_logits_per_layer[-1]  # (B, Q, num_classes+1)

        # RbA: per ogni query, prob di essere una classe ID (esclude void/last class)
        # class_logits shape: (B, Q, C+1) — l'ultima classe è void/"no object"
        class_probs = torch.softmax(class_logits, dim=-1)  # (B, Q, C+1)
        id_probs    = class_probs[..., :-1].sum(dim=-1)    # (B, Q) — prob ID per query

        # mask_logits: (B, Q, H, W) — sigmoid per probabilità di appartenenza
        mask_probs  = torch.sigmoid(mask_logits)           # (B, Q, H, W)

        # RbA score per pixel: max su Q di (mask_prob * id_prob)
        id_probs    = id_probs[..., None, None]            # (B, Q, 1, 1)
        rba_score   = (mask_probs * id_probs).max(dim=1)[0]  # (B, H, W)

        rba_score   = model.revert_window_logits_semantic(
            rba_score.unsqueeze(1), origins, img_sizes
        )[0][0]  # (H, W)

    # anomaly = bassa appartenenza a qualsiasi maschera ID
    anomaly_rba = (1.0 - rba_score.float()).cpu().numpy()
    return anomaly_rba

# ── GT remapping ──────────────────────────────────────────────────────────────

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
        ood_gts = np.where(ood_gts < 20,  0,   ood_gts)
        ood_gts = np.where(ood_gts == 255, 1,  ood_gts)
    return ood_gts


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/content/drive/MyDrive/Anomaly_Segmentation_Datasets/Validation_Dataset/RoadAnomaly21/images/*.png",
        nargs="+",
        help="Glob pattern for input images, e.g. 'path/to/images/*.png'",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--config",
        default="../configs/dinov2/cityscapes/semantic/eomt_base_640.yaml",
        help="Path to the EoMT YAML config file",
    )
    parser.add_argument(
        "--img_size", type=int, nargs=2, default=None,
        help="Override inference image size as H W, e.g. --img_size 640 640 (default: 640 640 from config)",
    )
    parser.add_argument(
        "--num_classes", type=int, default=None,
        help="Override number of classes (default: 19 for Cityscapes)",
    )
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = "cpu" if args.cpu else "cuda"

    # ── load model ────────────────────────────────────────────────────────────
    print(f"Loading EoMT from config: {args.config}")
    img_size    = tuple(args.img_size) if args.img_size else None
    model, img_size = load_model(args.config, device, img_size=img_size, num_classes=args.num_classes)
    print(f"Model loaded. Inference image size: {img_size}")

    # ── preprocessing ────────────────────────────────────────────────────────
    # NOTE: do NOT use ToTensor() here — window_imgs_semantic expects a uint8
    # PIL image or tensor in (C, H, W) uint8 format, not float32.
    # We only resize; EoMT handles normalisation internally.
    input_transform = Resize(img_size, Image.BILINEAR)
    target_transform = Compose([
        Resize(img_size, Image.NEAREST),
    ])

    # ── results file ──────────────────────────────────────────────────────────
    if not os.path.exists("results.txt"):
        open("results.txt", "w").close()
    result_file = open("results.txt", "a")

    # ── image loop ────────────────────────────────────────────────────────────
    anomaly_score_list         = []
    anomaly_score_list_logit   = []
    anomaly_score_list_entropy = []
    anomaly_score_list_rba     = []
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

        # keep as PIL image after resize — window_imgs_semantic handles the rest
        pil_img = input_transform(Image.open(path).convert("RGB"))
        # convert to uint8 tensor (C, H, W) as expected by window_imgs_semantic
        img_tensor = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1)  # uint8 (C,H,W)

        anomaly_msp, anomaly_logit, anomaly_entropy = infer_single(
            model, img_tensor, img_size, device, temperature=args.temperature
        )
        anomaly_rba = infer_single_rba(model, img_tensor, img_size, device)
        
        # ground-truth mask path
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

        print(f"  GT:     {pathGT}")
        print(f"  Values: {np.unique(ood_gts)}")

        if 1 not in np.unique(ood_gts):
            print("  No OOD pixels – skipping.")
            continue

        ood_gts_list.append(ood_gts)
        anomaly_score_list.append(anomaly_msp)
        anomaly_score_list_logit.append(anomaly_logit)
        anomaly_score_list_entropy.append(anomaly_entropy)
        anomaly_score_list_rba.append(anomaly_rba)

        torch.cuda.empty_cache()

    if len(ood_gts_list) == 0:
        print("No valid images processed. Check dataset paths and GT masks.")
        result_file.close()
        return

    # ── aggregate ─────────────────────────────────────────────────────────────
    ood_gts                = np.array(ood_gts_list)               # (N, H, W)
    anomaly_scores         = np.array(anomaly_score_list)          # (N, H, W)
    anomaly_scores_logit   = np.array(anomaly_score_list_logit)
    anomaly_scores_entropy = np.array(anomaly_score_list_entropy)

    valid_mask = (ood_gts != IGNORE_INDEX)
    ood_mask   = (ood_gts == 1) & valid_mask
    ind_mask   = (ood_gts == 0) & valid_mask

    # ── metrics ───────────────────────────────────────────────────────────────
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
    auprc_rba, fpr_rba         = compute_metrics(np.array(anomaly_score_list_rba), ood_mask, ind_mask, "RbA")

    result_file.write(
        f"  AUPRC (MSP): {auprc_msp*100:.2f}%   FPR@TPR95 (MSP): {fpr_msp*100:.2f}%"
        f"  |  AUPRC (MaxLogit): {auprc_logit*100:.2f}%   FPR@TPR95 (MaxLogit): {fpr_logit*100:.2f}%"
        f"  |  AUPRC (Entropy): {auprc_entropy*100:.2f}%   FPR@TPR95 (Entropy): {fpr_entropy*100:.2f}%\n"
    )
    result_file.close()


if __name__ == "__main__":
    main()


