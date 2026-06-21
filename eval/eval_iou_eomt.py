# BOZZA BASATA SU EVAL_IOU 

import os
import sys

# Add EoMT repo root to path so that 'datasets', 'models', 'training' modules are importable
sys.path.insert(0, "/content/MaskArchitectureAnomaly_CourseProject/eomt")

import glob
import time
import yaml
import torch
import random
import warnings
import importlib
import torch.nn.functional as F
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from torch.amp.autocast_mode import autocast
from torchvision.transforms import Resize
from iouEval import iouEval, getColorEntry

# reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

IGNORE_INDEX = 255
IOU_IGNORE_CLASS = 19 



# Model loading 

def load_model(config_path: str, device: str, img_size=None, num_classes=None, weights_path=None):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_kwargs = config["data"].get("init_args", {})

    # img_size and num_classes: use explicit values if provided, otherwise read from config
    if img_size is None:
        img_size = data_kwargs.get("img_size", (1024, 1024))
    img_size = tuple(img_size)

    if num_classes is None:
        num_classes = data_kwargs.get("num_classes", 19)

    print(f"  img_size={img_size}  num_classes={num_classes}")

    # build the encoder
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

    # build the lightning module
    lit_mod, lit_cls_name = config["model"]["class_path"].rsplit(".", 1)
    lit_cls = getattr(importlib.import_module(lit_mod), lit_cls_name)
    model_kwargs = {k: v for k, v in config["model"]["init_args"].items() if k != "network"}

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

    # weights explicit --weights argument takes priority over the default path which is pretrained ones
    if weights_path is None:
        weights_path = "/content/drive/MyDrive/eomt_weights/eomt_cityscapes.bin"

    print(f"Loading weights from: {weights_path}")
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)

    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys ({len(missing_keys)}): {missing_keys[:10]}")
    print(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:10]}")
    print("Weights loaded successfully.")

    return model, img_size



# Semantic-segmentation inference: same pixel_probs combination used for
# MSP/MaxLogit/Entropy, but we only need the argmax class here.

def infer_semantic(model, img_tensor: torch.Tensor, img_size, device: str) -> torch.Tensor:
    """
    Runs EoMT on a single image and returns the predicted semantic map (H, W)
    as a LongTensor with class ids in [0, num_classes-1] (void/no-object excluded,
    exactly as for the anomaly-score baselines).
    """
    dtype = torch.float16 if device != "cpu" else torch.float32

    with torch.no_grad(), autocast(dtype=dtype, device_type=device):
        imgs = [img_tensor.to(device)]
        img_sizes = [img_tensor.shape[-2:]]

        # sliding-window preprocessing (official EoMT pipeline)
        crops, origins = model.window_imgs_semantic(imgs)

        mask_logits_per_layer, class_logits_per_layer = model(crops)

        # use last decoder layer (best quality)
        mask_logits = F.interpolate(
            mask_logits_per_layer[-1], img_size, mode="bilinear", align_corners=False
        )
        class_logits = class_logits_per_layer[-1]  # (B, Q, num_classes+1)

        mask_logits = model.revert_window_logits_semantic(mask_logits, origins, img_sizes)[0]
        class_logits = class_logits[0]  # (Q, num_classes+1)

        mask_logits = mask_logits.float()
        class_logits = class_logits.float()

    mask_probs = torch.sigmoid(mask_logits)                              # [Q, H, W]
    class_probs_full = torch.softmax(class_logits, dim=-1)               # [Q, C+1]
    class_probs_id = class_probs_full[:, :-1]                            # [Q, C]  (drop void)
    pixel_probs = torch.einsum("qc,qhw->chw", class_probs_id, mask_probs)  # [C, H, W]

    pred = torch.argmax(pixel_probs, dim=0).cpu()                        # [H, W], values in [0, C-1]
    return pred



# Minimal Cityscapes val dataset (image / labelTrainIds pairs).
# may be replaced with the actual dataloader

def build_cityscapes_filelist(datadir: str, subset: str):
    img_pattern = os.path.join(datadir, "leftImg8bit", subset, "*", "*_leftImg8bit.png")
    img_paths = sorted(glob.glob(img_pattern))

    pairs = []
    for img_path in img_paths:
        label_path = (
            img_path
            .replace("leftImg8bit", "gtFine", 1)
            .replace("_leftImg8bit.png", "_gtFine_labelTrainIds.png")
        )
        if os.path.exists(label_path):
            pairs.append((img_path, label_path))
        else:
            print(f"  WARNING: missing label for {img_path}, skipping.")
    return pairs


# MAIN

def main():
    parser = ArgumentParser()
    parser.add_argument("--datadir", default=os.getenv("HOME") + "/datasets/cityscapes/")
    parser.add_argument("--subset", default="val")  # val or train (must have labelTrainIds)
    parser.add_argument(
        "--config",
        default="../configs/dinov2/cityscapes/semantic/eomt_base_640.yaml",
        help="Path to the EoMT YAML config file",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Path to the checkpoint to evaluate (pretrained / eim / oe / rba / ...).",
    )
    parser.add_argument("--img_size", type=int, nargs=2, default=None)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = "cpu" if args.cpu else "cuda"

    print(f"Loading EoMT from config: {args.config}")
    img_size_arg = tuple(args.img_size) if args.img_size else None
    model, img_size = load_model(
        args.config, device, img_size=img_size_arg, num_classes=args.num_classes, weights_path=args.weights
    )
    print(f"Model loaded. Inference image size: {img_size}")

    # NOTE: no normalization here, EoMT applies pixel_mean/pixel_std internally, only resize is needed
    input_transform = Resize(img_size, Image.BILINEAR)
    target_transform = Resize(img_size, Image.NEAREST)

    pairs = build_cityscapes_filelist(args.datadir, args.subset)
    if not pairs:
        print(f"Error: no image/label pairs found under {args.datadir} (subset={args.subset})")
        return
    print(f"Found {len(pairs)} image/label pairs.")

    # same trick used in ERFNet's eval_iou.py: add a dummy 20th class for the
    # ignore label, so the IoU implementation can be reused unmodified.
    iouEvalVal = iouEval(IOU_IGNORE_CLASS + 1)

    start = time.time()
    for step, (img_path, label_path) in enumerate(pairs):
        pil_img = input_transform(Image.open(img_path).convert("RGB"))
        img_tensor = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1)  # uint8 (C,H,W)

        pred = infer_semantic(model, img_tensor, img_size, device)  # (H, W), long, in [0, num_classes-1]

        label_img = target_transform(Image.open(label_path))
        label = torch.from_numpy(np.array(label_img)).long()  # (H, W), 0..18 + 255
        label[label == IGNORE_INDEX] = IOU_IGNORE_CLASS         # 255 -> 19, like ERFNet's Relabel(255, 19)

        iouEvalVal.addBatch(
            pred.unsqueeze(0).unsqueeze(0),
            label.unsqueeze(0).unsqueeze(0),
        )

        print(step, os.path.basename(img_path))
        torch.cuda.empty_cache()

    iouVal, iou_classes = iouEvalVal.getIoU()

    class_names = [
        "Road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
        "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
        "truck", "bus", "train", "motorcycle", "bicycle",
    ]

    print("---------------------------------------")
    print("Took ", time.time() - start, "seconds")
    print("=======================================")
    print("Per-Class IoU:")
    for i, name in enumerate(class_names):
        iouStr = getColorEntry(iou_classes[i]) + "{:0.2f}".format(iou_classes[i] * 100) + "\033[0m"
        print(iouStr, name)
    print("=======================================")
    iouStr = getColorEntry(iouVal) + "{:0.2f}".format(iouVal * 100) + "\033[0m"
    print("MEAN IoU: ", iouStr, "%")


if __name__ == "__main__":
    main()
