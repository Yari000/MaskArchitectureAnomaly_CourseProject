# CLASSE PER OUTLIER EXPOSURE 
# Wrapper attorno a CityscapesSemantic che incolla patch OOD da COCO
# sulle immagini di training. La outlier_mask viene aggiunta al dict_state
# target come chiave "outlier_mask" (boolean, HxW)
# Codice adattato al lightning-module

import random
import json
import zipfile
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from pycocotools import mask as coco_mask_util

from datasets.lightning_data_module import LightningDataModule
from datasets.cityscapes_semantic import CityscapesSemantic


# Categorie COCO con overlap rispetto a Cityscapes vengono ESCLUSE
_CITYSCAPES_OVERLAP = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "traffic light", "stop sign", "parking meter", "bench",
}


class CityscapesCOCOPaste(LightningDataModule):
    """
    CityscapesSemantic + COCO Cut-and-Paste per Outlier Exposure.

    Args:
        path: path Cityscapes (stesso formato di Cityscapes_Semantic)
        coco_img_dir: cartella con le immagini COCO (... val2017/)
        coco_ann_file: path al JSON delle istanze COCO
                       (es. annotations/instances_val2017.json)
        paste_prob: probabilità di incollare patch su un'immagine
        num_patches_range: (min, max) patch per immagine
        min_area: area minima dell'annotazione COCO in pixel^2
        paste_scale_range: (min, max) scala della patch relativa
                           all'altezza dell'immagine target
    """

    def __init__(
        self,
        path,
        coco_img_dir: str,
        coco_ann_file: str,
        num_workers: int = 4,
        batch_size: int = 16,
        img_size: tuple[int, int] = (1024, 1024),
        num_classes: int = 19,
        color_jitter_enabled: bool = True,
        scale_range: tuple[float, float] = (0.5, 2.0),
        paste_prob: float = 0.5,
        num_patches_range: tuple[int, int] = (1, 3),
        min_area: int = 1000,
        paste_scale_range: tuple[float, float] = (0.05, 0.2),
    ) -> None:
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=img_size,
            check_empty_targets=True,
        )
        self.save_hyperparameters(ignore=["_class_path"])

        # Instanziamo CityscapesSemantic interno per riutilizzare setup/transforms
        self._cityscapes = CityscapesSemantic(
            path=path,
            num_workers=num_workers,
            batch_size=batch_size,
            img_size=img_size,
            num_classes=num_classes,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
        )

        self.coco_img_dir = Path(coco_img_dir)
        self.coco_ann_file = Path(coco_ann_file)
        self.paste_prob = paste_prob
        self.num_patches_range = num_patches_range
        self.min_area = min_area
        self.paste_scale_range = paste_scale_range

        # Carica annotazioni COCO e filtra categorie OOD
        self._ann_list = None  # lazy init

    
    # Lazy init delle annotazioni COCO 
    
    def _ensure_ann_list(self):
        if self._ann_list is not None:
            return

        with open(self.coco_ann_file, "r") as f:
            data = json.load(f)

        # Mappa category_id -> name
        cat_id_to_name = {c["id"]: c["name"] for c in data["categories"]}

        # Mappa image_id -> file_name
        self._img_id_to_file = {
            img["id"]: img["file_name"] for img in data["images"]
        }

        # Filtra annotazioni: solo OOD, non crowd, area sufficiente
        self._ann_list = [
            ann for ann in data["annotations"]
            if not ann["iscrowd"]
            and ann["area"] >= self.min_area
            and cat_id_to_name.get(ann["category_id"], "") not in _CITYSCAPES_OVERLAP
        ]

   
    # Estrae una patch (rgb numpy HxWx3, mask binaria HxW) da COCO
    
    def _get_random_patch(self):
        self._ensure_ann_list()

        ann = random.choice(self._ann_list)
        file_name = self._img_id_to_file[ann["image_id"]]
        img_path = self.coco_img_dir / file_name

        img = np.array(Image.open(img_path).convert("RGB"))
        H_img, W_img = img.shape[:2]

        # Decodifica maschera RLE o poligoni
        seg = ann["segmentation"]
        if isinstance(seg, dict):  # RLE
            rle = seg
        else:  # lista di poligoni
            rle = coco_mask_util.frPyObjects(seg, H_img, W_img)
            rle = coco_mask_util.merge(rle)
        binary_mask = coco_mask_util.decode(rle).astype(bool)  # HxW

        # Ritaglia sul bounding box
        x, y, w, h = [int(v) for v in ann["bbox"]]
        x2, y2 = min(x + w, W_img), min(y + h, H_img)
        patch_img = img[y:y2, x:x2]
        patch_mask = binary_mask[y:y2, x:x2]

        return patch_img, patch_mask

    
    # Incolla N patch COCO su img_tensor, aggiorna outlier_mask
    # img_tensor: tv_tensors.Image (C, H, W), uint8
   
    def _paste_coco_patches(self, img_tensor):
        # Converti in numpy HxWxC per cv2
        img_np = img_tensor.permute(1, 2, 0).numpy().copy()
        H, W = img_np.shape[:2]
        outlier_mask = np.zeros((H, W), dtype=bool)

        n = random.randint(*self.num_patches_range)
        for _ in range(n):
            try:
                patch_img, patch_mask = self._get_random_patch()
            except Exception:
                continue

            # Scala la patch
            scale = random.uniform(*self.paste_scale_range)
            new_h = max(1, int(H * scale))
            orig_h, orig_w = patch_img.shape[:2]
            if orig_h == 0 or orig_w == 0:
                continue
            new_w = max(1, int(orig_w * new_h / orig_h))

            patch_img = cv2.resize(patch_img, (new_w, new_h))
            patch_mask = cv2.resize(
                patch_mask.astype(np.uint8), (new_w, new_h),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

            # Posizione random
            if H - new_h <= 0 or W - new_w <= 0:
                continue
            top = random.randint(0, H - new_h)
            left = random.randint(0, W - new_w)

            # Incolla solo sui pixel della maschera
            roi = img_np[top:top + new_h, left:left + new_w]
            roi[patch_mask] = patch_img[patch_mask]
            img_np[top:top + new_h, left:left + new_w] = roi
            outlier_mask[top:top + new_h, left:left + new_w] |= patch_mask

        img_out = torch.from_numpy(img_np).permute(2, 0, 1)
        outlier_tensor = torch.from_numpy(outlier_mask)  # bool HxW
        return img_out, outlier_tensor

   
    # Wrapper dataset interno
   
    class _PasteDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset, paste_fn, paste_prob):
            self.base = base_dataset
            self.paste_fn = paste_fn
            self.paste_prob = paste_prob

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            img, target = self.base[idx]
            H, W = img.shape[-2:]

            if random.random() < self.paste_prob:
                img, outlier_mask = self.paste_fn(img)
            else:
                outlier_mask = torch.zeros(H, W, dtype=torch.bool)

            # Aggiunge outlier_mask al dict target 
            target["outlier_mask"] = outlier_mask
            return img, target

   
    # LightningDataModule interface, omologato agli altri file in _datasets
   
    def setup(self, stage: Union[str, None] = None):
        self._cityscapes.setup(stage)

        self.train_dataset = self._PasteDataset(
            base_dataset=self._cityscapes.cityscapes_train_dataset,
            paste_fn=self._paste_coco_patches,
            paste_prob=self.paste_prob,
        )
        self.val_dataset = self._cityscapes.cityscapes_val_dataset
        return self

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )
