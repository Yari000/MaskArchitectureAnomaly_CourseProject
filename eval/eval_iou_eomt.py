# FILE BASATO SU EVAL_IOU 

import os
import sys

# Add EoMT repo root to path so that 'datasets', 'models', 'training' modules are importable
sys.path.insert(0, "/content/MaskArchitectureAnomaly_CourseProject/eomt")

import glob
import time
import yaml
import torch
import io
import zipfile
import random
import warnings
import importlib
import torch.nn.functional as F
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from torch.amp.autocast_mode import autocast
from torchvision.transforms import Resize
from torchvision.datasets import Cityscapes
from iouEval import iouEval, getColorEntry

# reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

IGNORE_INDEX = 255
NUM_CLASSES   = 19
IOU_N_CLASSES = 20  # 0-18 semantic + 19 for ignore, same convention as ERFNet eval_iou.py

# labelId -> trainId mapping, built from torchvision.datasets.Cityscapes
# mirrors the target_parser in CityscapesSemantic 

# path to cityscapes dataset
img_zip = "/content/drive/MyDrive/Anomaly_Segmentation_Datasets/Cityscapes/leftImg8bit_trainvaltest.zip"

with zipfile.ZipFile(img_zip, "r") as z:
    # show first names to debug
    names = z.namelist()
    print("Totale entries:", len(names))
    print("Prime 10 entries:")
    for n in names[:10]:
        print(" ", repr(n))
    
    # prova a leggere la prima entry che sembra un'immagine
    img_entries = [n for n in names if n.endswith("_leftImg8bit.png")]
    print(f"\nTrovate {len(img_entries)} immagini")
    if img_entries:
        first = img_entries[0]
        print("Try reading:", first)
        info = z.getinfo(first)
        print("  compress_type:", info.compress_type)
        print("  file_size:", info.file_size)
        with z.open(first) as f:
            data = f.read()
        print("  bytes read:", len(data))
        # se bytes > 0, proviamo ad aprire
        if len(data) > 0:
            from PIL import Image
            img = Image.open(io.BytesIO(data))
            print("  PIL mode:", img.mode, "size:", img.size)
        else:
            print("  Errore: 0 bytes letti")




def build_id_to_trainid() -> np.ndarray:
    """Returns a 256-element array: label_id -> train_id (255 = ignore)."""
    mapping = np.full(256, IGNORE_INDEX, dtype=np.uint8)
    for cls in Cityscapes.classes:
        if cls.ignore_in_eval:
            mapping[cls.id] = IGNORE_INDEX
        else:
            mapping[cls.id] = cls.train_id
    return mapping

# Cityscapes val loader that reads directly from the two zip files

def build_zip_pairs(img_zip_path: str, gt_zip_path: str, subset: str):
    """
    Scan the image zip and find matching label files in the GT zip.
    Returns a list of (img_zip_name, gt_zip_name) internal paths.
    """
    print(f"Scanning image zip: {img_zip_path}")
    with zipfile.ZipFile(img_zip_path, "r") as zimg:
          all_img = [n for n in zimg.namelist()
           if f"/leftImg8bit/{subset}/" in n 
           and n.endswith("_leftImg8bit.png")
           and not n.startswith("__MACOSX")]
                   
    all_img.sort()
    print(f"  Found {len(all_img)} images for subset='{subset}'")

    print(f"Scanning GT zip:    {gt_zip_path}")
    with zipfile.ZipFile(gt_zip_path, "r") as zgt:
        gt_names = set(n for n in zgt.namelist() if not n.startswith("__MACOSX"))

    pairs = []
    missing = 0
    for img_name in all_img:
        # derive matching label path
        # e.g. leftImg8bit_trainvaltest/leftImg8bit/val/aachen/aachen_000000_000019_leftImg8bit.png
        #   -> gtFine_trainvaltest/gtFine/val/aachen/aachen_000000_000019_gtFine_labelIds.png
        gt_name = (img_name
                   .replace("leftImg8bit_trainvaltest/leftImg8bit/", "gtFine_trainvaltest/gtFine/")
                   .replace("_leftImg8bit.png", "_gtFine_labelIds.png"))
        if gt_name in gt_names:
            pairs.append((img_name, gt_name))
        else:
            print(f"  WARNING: no label found for {img_name}, skipping.")
            missing += 1

    print(f"  Matched {len(pairs)} pairs ({missing} skipped).")
    return pairs


def read_image_from_zip(zf: zipfile.ZipFile, name: str) -> Image.Image:
    with zf.open(name) as f:
        return Image.open(io.BytesIO(f.read())).convert("RGB")


def read_label_from_zip(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    with zf.open(name) as f:
        return np.array(Image.open(io.BytesIO(f.read())).convert("L"))


# Model loading as in EoMT evalAnomaly

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
        masked_attn_enabled=False,  # always disabled at inference
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
        imgs      = [img_tensor.to(device)]
        img_sizes = [img_tensor.shape[-2:]]
        crops, origins = model.window_imgs_semantic(imgs)
        mask_logits_per_layer, class_logits_per_layer = model(crops)
        mask_logits = F.interpolate(
            mask_logits_per_layer[-1], img_size, mode="bilinear", align_corners=False
        )
        crop_logits = model.to_per_pixel_logits_semantic(
            mask_logits, class_logits_per_layer[-1]
        )
        logits = model.revert_window_logits_semantic(crop_logits, origins, img_sizes)
        pixel_logits = logits[0].float().cpu()  # [C, H, W]

    pred = torch.argmax(pixel_logits, dim=0)  # [H, W]
    return pred

# MAIN----------------------------------

def main():
    
    parser = ArgumentParser()
    parser.add_argument(
        "--datadir",
        default="/content/drive/MyDrive/Anomaly_Segmentation_Datasets/Cityscapes/",
        help="Folder that contains leftImg8bit_trainvaltest.zip and gtFine_trainvaltest.zip",
    )
    parser.add_argument("--subset", default="val")
    parser.add_argument(
        "--config",
        default="../configs/dinov2/cityscapes/semantic/eomt_base_640.yaml",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Checkpoint path (pretrained / eim / oe / rba / ...). "
             "Defaults to the hardcoded path in load_model.",
    )
    parser.add_argument("--img_size", type=int, nargs=2, default=None)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = "cpu" if args.cpu else "cuda"

    # model 

    print(f"Loading EoMT from config: {args.config}")
    img_size_arg = tuple(args.img_size) if args.img_size else None
    model, img_size = load_model(
        args.config, device, img_size=img_size_arg, num_classes=args.num_classes, weights_path=args.weights
    )
    print(f"Model loaded. Inference image size: {img_size}")

    # label remapping table: labelId -> trainId 
    id_to_trainid = build_id_to_trainid()  # shape (256,), uint8

    # zip paths 
    img_zip = os.path.join(args.datadir, "leftImg8bit_trainvaltest.zip")
    gt_zip  = os.path.join(args.datadir, "gtFine_trainvaltest.zip")

    for p in [img_zip, gt_zip]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Expected zip not found: {p}")

    pairs = build_zip_pairs(img_zip, gt_zip, args.subset)
    if not pairs:
        print("No pairs found. Check subset name and zip contents.")
        return

    # image resize (label resize uses NEAREST, applied per-image below)
    img_resize   = Resize(img_size, Image.BILINEAR)
    label_resize = Resize(img_size, Image.NEAREST)

    iouEvalVal = iouEval(IOU_N_CLASSES)

    start = time.time()
    with zipfile.ZipFile(img_zip, "r") as zimg, zipfile.ZipFile(gt_zip, "r") as zgt:
        for step, (img_name, gt_name) in enumerate(pairs):

            # image
            pil_img    = img_resize(read_image_from_zip(zimg, img_name))
            img_tensor = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1)  # uint8 (C,H,W)

            # prediction 
            pred = infer_semantic(model, img_tensor, img_size, device)  # (H,W) long, 0..18

            # label: labelId -> trainId -> resize -> ignore-remap 
            raw_label   = read_label_from_zip(zgt, gt_name)             # (H,W) uint8, labelIds
            train_label = id_to_trainid[raw_label]                      # (H,W) uint8, trainIds + 255
            label_pil   = label_resize(Image.fromarray(train_label))
            label_np    = np.array(label_pil).astype(np.int64)
            # remap 255 (ignore) -> 19, same trick as ERFNet eval_iou.py Relabel(255,19)
            label_np[label_np == IGNORE_INDEX] = NUM_CLASSES             # 255 -> 19

            label_tensor = torch.from_numpy(label_np)

            iouEvalVal.addBatch(
                pred.unsqueeze(0).unsqueeze(0),
                label_tensor.unsqueeze(0).unsqueeze(0),
            )

            print(step, os.path.basename(img_name))
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
