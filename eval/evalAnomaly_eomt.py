
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

# reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

IGNORE_INDEX = 255


# loading the model

def load_model(config_path: str, device: str, img_size=None, num_classes=None):
    """
    Instantiate and load an EoMT Lightning model from a YAML config.
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
        img_size = data_kwargs.get("img_size", (1024, 1024))  #1024?
    
    # FIX (debug)
    img_size = tuple(img_size)
    
    if num_classes is None:
        num_classes = data_kwargs.get("num_classes", 19)

    print(f"  img_size={img_size}  num_classes={num_classes}")

    # build the econder
    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    enc_mod, enc_cls_name = encoder_cfg["class_path"].rsplit(".", 1)
    encoder_cls = getattr(importlib.import_module(enc_mod), enc_cls_name)
    encoder = encoder_cls(img_size=img_size, **encoder_cfg.get("init_args", {}))

    # build the network (eomt)
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

    # build the lightninghface module
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

    # download the weights from the HuggingFace Hub
    name = (config.get("trainer", {})
                  .get("logger", {})
                  .get("init_args", {})
                  .get("name"))

    if name is None:
        warnings.warn("No logger name found in config — proceeding with random weights.")
        return model, img_size

    # loading the weights from the local checkpoint
    ckpt_path = "/content/drive/MyDrive/eomt_weights/eomt_cityscapes.bin"
    print(f"Loading weights from: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)

    # lightning checkpoints wrap the weights under the state_dict variable
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # Check to see if weights are passed correctly
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys ({len(missing_keys)}): {missing_keys[:10]}")
    print(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:10]}")
    print("Weights loaded successfully.")

    return model, img_size


# inference for a single image

def infer_single(model, img_tensor: torch.Tensor, img_size, device: str):
    """
    Run EoMT semantic inference on a single image tensor (C, H, W).
    NOTE: no external normalization, EoMT applies pixel_mean/pixel_std internally.

    Returns three (H, W) float32 numpy arrays:
        anomaly_msp     – 1 - max softmax probability  (MSP)
        anomaly_logit   – negative max logit            (MaxLogit)
        anomaly_entropy – predictive entropy            (Entropy)
    """
    dtype = torch.float16 if device != "cpu" else torch.float32

    with torch.no_grad(), autocast(dtype=dtype, device_type=device):
        imgs = [img_tensor.to(device)]
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

        # before computing probs return to the original image dimension :(q,h,w)
        mask_logits = model.revert_window_logits_semantic(mask_logits, origins, img_sizes)[0] 
        class_logits = class_logits[0]  # (q, numclasses+1)

        # FIX
        # convert to float to avoid underflow
        mask_logits = mask_logits.float()
        class_logits = class_logits.float() 

    # take only known classes (in-distribution ones) excluding the last column
    id_class_logits = class_logits[:, :-1]  # Shape: (Q, num_classes)

    # transform spacial query logits into binary activation probs thru a sigmoid
    mask_probs = torch.sigmoid(mask_logits)  # Shape: (Q, H, W)

    # Evaluate prob distribution over known classes
    class_probs = torch.softmax(id_class_logits, dim=-1) # shape (q, numclasses)

    # spacial projection to get P(class|query)*P(query|pixel) = P(class|pixel)
    pixel_probs = torch.einsum("qc,qhw->chw", class_probs, mask_probs)      # Shape: (num_classes, H, W)
    pixel_logits = torch.einsum("qc,qhw->chw", id_class_logits, mask_probs)  # Shape: (num_classes, H, W)

    # evaluate the metrics
    anomaly_msp = (1.0 - torch.max(pixel_probs, dim=0)[0]).cpu().numpy()
    anomaly_logit = (-torch.max(pixel_logits, dim=0)[0]).cpu().numpy()

    # probabilities are normalized to guarantee a valid prob distribution to evaluate entropy
    pixel_probs_norm = pixel_probs / (pixel_probs.sum(dim=0, keepdim=True) + 1e-8)
    anomaly_entropy = (-torch.sum(pixel_probs_norm * torch.log(pixel_probs_norm + 1e-8), dim=0)).cpu().numpy()

    # return raw logits to evaluate temp scaling baseline outside of the loop
    return anomaly_msp, anomaly_logit, anomaly_entropy

# rejected by all inference

def infer_single_rba(model, img_tensor, img_size, device):
    """
    inference over the RbA method (rejected by all): a pixel is considered an anomaly if its rejected by 
    all queries that show high in-distribution confidence
    """
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

        # RbA: for every query is the prob of being into a ID class (excluding void/last classes)
        # class_logits shape: (B, Q, C+1) — last class is VOID
        # class_probs = torch.softmax(class_logits, dim=-1)  # (B, Q, C+1)
        # id_probs    = class_probs[..., :-1].sum(dim=-1)    # (B, Q) — ID prob for the query

        # retrieve spacial geometry by getting toghether the crops
        mask_logits = model.revert_window_logits_semantic(mask_logits, origins, img_sizes)[0]  # (Q, H, W)
        class_logits = class_logits[0]  # (Q, num_classes + 1)
        mask_logits = mask_logits.float()
        class_logits = class_logits.float()

        # evaluate prob that every query belongs to a known class
        class_probs = torch.softmax(class_logits, dim=-1)  # (Q, num_classes + 1)
        id_probs = class_probs[:, :-1].sum(dim=-1)         # Shape: (Q,)

        # mask_logits: (B, Q, H, W) — sigmoid per probabilità di appartenenza
        mask_probs  = torch.sigmoid(mask_logits)           # (B, Q, H, W)

        # expand dimensions to abilitate pixel-to-pixel broadcasting
        id_probs = id_probs[:, None, None]  # Shape: (Q, 1, 1)

        # RbA score per pixel: max su Q di (mask_prob * id_prob)
        #id_probs    = id_probs[..., None, None]            # (B, Q, 1, 1)
        #rba_score   = (mask_probs * id_probs).max(dim=1)[0]  # (B, H, W)

        # evaluate probs that a pixel DOES NOT belong to a valid class for the q-th query
        rejected_by_query = 1.0 - (mask_probs * id_probs)  # Shape: (Q, H, W)

    # anomaly = bassa appartenenza a qualsiasi maschera ID
    anomaly_rba = torch.prod(rejected_by_query, dim=0).cpu().numpy()  # Shape: (H, W)
    return anomaly_rba

# GT remapping

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


# main

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/content/drive/MyDrive/Anomaly_Segmentation_Datasets/Validation_Dataset/RoadAnomaly21/images/*.png",
        nargs="+",
        help="Glob pattern for input images, e.g. 'path/to/images/*.png'",
    )
    parser.add_argument("--temperature", type=float, nargs="+", default=[0.5, 0.75, 1.1, 1.25, 1.5])
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

    # load the model
    print(f"Loading EoMT from config: {args.config}")
    img_size    = tuple(args.img_size) if args.img_size else None
    model, img_size = load_model(args.config, device, img_size=img_size, num_classes=args.num_classes)
    print(f"Model loaded. Inference image size: {img_size}")

    # image preprocessing
    
    # NOTE: do NOT use ToTensor() here, window_imgs_semantic expects a uint8
    # PIL image or tensor in (C, H, W) uint8 format, not float32.
    # We only resize; EoMT handles normalisation internally.
    input_transform = Resize(img_size, Image.BILINEAR)
    target_transform = Compose([
        Resize(img_size, Image.NEAREST),
    ])

    # results file
    if not os.path.exists("results.txt"):
        open("results.txt", "w").close()
    result_file = open("results.txt", "a")

    # image inference loop
    anomaly_score_list         = []
    anomaly_score_list_logit   = []
    anomaly_score_list_temp    = {t: [] for t in args.temperature}
    anomaly_score_list_entropy = []
    anomaly_score_list_rba     = []
    ood_gts_list               = []
    class_logit_list           = []
    mask_probs_list            = []

    pattern = os.path.expanduser(str(args.input[0]))
    paths = sorted(glob.glob(pattern))

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

        # keep as PIL image after resize, window_imgs_semantic handles the rest
        pil_img = input_transform(Image.open(path).convert("RGB"))
        # convert to uint8 tensor (C, H, W) as expected by window_imgs_semantic
        img_tensor = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1)  # uint8 (C,H,W)

        anomaly_msp, anomaly_logit, anomaly_entropy, class_logit, mask_probs = infer_single(
            model, img_tensor, img_size, device
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
        class_logit_list.append(class_logit)
        mask_probs_list.append(mask_probs)

        torch.cuda.empty_cache()

    if len(ood_gts_list) == 0:
        print("No valid images processed. Check dataset paths and GT masks.")
        result_file.close()
        return

    # aggregate
    ood_gts                = np.array(ood_gts_list)               # (N, H, W)
    anomaly_scores         = np.array(anomaly_score_list)          # (N, H, W)
    anomaly_scores_logit   = np.array(anomaly_score_list_logit)
    anomaly_scores_entropy = np.array(anomaly_score_list_entropy)

    valid_mask = (ood_gts != IGNORE_INDEX)
    ood_mask   = (ood_gts == 1) & valid_mask
    ind_mask   = (ood_gts == 0) & valid_mask

    # computing the metrics (AUPCR and FPR)
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
    aupcr_temp = []
    fpr_temp = []

    # compute temp scaling in range --temperature
    for i, t in enumerate(args.temperature):
        anomaly_msp_t, anomaly_entropy_t = [], []
        for cl, mp in zip(class_logit_list, mask_probs_list):
            id_logits_t = cl[:, :-1] / t
            class_probs_t = torch.softmax(id_logits_t, dim=-1)
            pixel_probs_t = torch.einsum("qc,qhw->chw", class_probs_t, mp)
        
            # MSP
            anomaly_msp_t.append((1.0 - torch.max(pixel_probs_t, dim=0)[0]).numpy())
        
            # Entropy
            pixel_probs_norm = pixel_probs_t / (pixel_probs_t.sum(dim=0, keepdim=True) + 1e-8)
            anomaly_entropy_t.append(
                (-torch.sum(pixel_probs_norm * torch.log(pixel_probs_norm + 1e-8), dim=0)).numpy()
        )
    
        compute_metrics(np.array(anomaly_msp_t), ood_mask, ind_mask, f"MSP T={t}")
        compute_metrics(np.array(anomaly_entropy_t), ood_mask, ind_mask, f"Entropy T={t}")

    result_file.write(
        f"  AUPRC (MSP): {auprc_msp*100:.2f}%   FPR@TPR95 (MSP): {fpr_msp*100:.2f}%"
        f"  |  AUPRC (MaxLogit): {auprc_logit*100:.2f}%   FPR@TPR95 (MaxLogit): {fpr_logit*100:.2f}%"
        f"  |  AUPRC (Entropy): {auprc_entropy*100:.2f}%   FPR@TPR95 (Entropy): {fpr_entropy*100:.2f}%\n"
    )
    result_file.close()


if __name__ == "__main__":
    main()


