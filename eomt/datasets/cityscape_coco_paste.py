# CLASSE PER OUTLIER EXPOSURE 
# Wrapper attorno a CityscapesSemantic che incolla patch OOD da COCO
# sulle immagini di training. La outlier_mask viene aggiunta al dict_state
# target come chiave "ood_mask" (boolean, HxW)
# Codice adattato al lightning-module

import random
import json
import zipfile
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
from torchvision import tv_tensors
from PIL import Image
from torch.utils.data import DataLoader
from pycocotools import mask as coco_mask_util

from datasets.lightning_data_module import LightningDataModule
from datasets.cityscapes_semantic import CityscapesSemantic


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

    # Categorie OOD esplicite, semanticamente distanti da Cityscapes per accentuare l'esposizione
    # un approccio più "conservativo" -escludere le sole categorie in comune- è stato considerato, ma non attuato vista 
    # l'impossibilità di allenare il modello per tante epoche su tante immagini cityscapes.
    
    _OOD_CATEGORIES = (
        "elephant", "giraffe", "zebra", "bear", "couch",
        "chair", "toaster", "microwave", "banana", "apple", "backpack",
    )

    def __init__(
        self,
        path,
        coco_img_dir: str,
        coco_ann_file: str,
        num_workers: int = 2,
        batch_size: int = 2,
        img_size: tuple[int, int] = (1024, 1024),
        num_classes: int = 19,
        color_jitter_enabled: bool = True,
        scale_range: tuple[float, float] = (0.5, 2.0),
        paste_prob: float = 0.25,
        num_patches_range: tuple[int, int] = (1, 3),
        min_area: int = 1000,
        paste_scale_range: tuple[float, float] = (0.08, 0.3),
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
        self._img_to_id_file = None

    
    # Lazy init delle annotazioni COCO 
    
    def _ensure_ann_list(self):
        if self._ann_list is not None:
            return

        with open(self.coco_ann_file, "r") as f:
            data = json.load(f)

        # Mappa category_id -> name
        cat_id_to_name = {c["id"]: c["name"] for c in data["categories"]}
        ood_names = set(self._OOD_CATEGORIES)

        # Mappa image_id -> file_name
        self._img_id_to_file = {
            img["id"]: img["file_name"] for img in data["images"]
        }

        # Filtro in positivo sulle categorie OOD esplicite
        self._ann_list = [
            ann for ann in data["annotations"]
            if not ann["iscrowd"]
            and ann["area"] >= self.min_area
            and cat_id_to_name.get(ann["category_id"], "") in ood_names
        ]

        if not self._ann_list:
            raise ValueError("Nessuna annotazione OOD trovata: verifica coco_ann_file e le categorie.")

       

   
    # Estrae una patch (rgb numpy HxWx3, mask binaria HxW) da COCO
    
    def _get_random_patch(self, max_tries: int = 20):
      self._ensure_ann_list()
      for _ in range(max_tries):   # prova un po' di volte
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
        x, y, w, h = map(int, ann["bbox"])
        x2= min(x+w, W_img)
        y2= min(y+h, H_img)
        patch_img = img[y:y2, x:x2]
        patch_mask = binary_mask[y:y2, x:x2]

        # evitiamo patch quasi vuote
        if patch_mask.sum() < 280:
            continue

        return patch_img, patch_mask
      raise RuntimeError("Unable to sample a valid patch")

    
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

            ph,pw = patch_img.shape[:2]

            # Scala la patch 
            scale = random.uniform(*self.paste_scale_range)
            h_target= int(H*scale)
            w_target= max(1, round(pw/ph*h_target))

            # non incolliamo patch troppo piccole
            if h_target < 8 or w_target < 8:
                continue

            patch_img = cv2.resize(patch_img, (w_target, h_target), interpolation=cv2.INTER_LINEAR)
            patch_mask = cv2.resize(
                patch_mask.astype(np.uint8), (w_target, h_target),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

            if patch_mask.sum() < 60:
                continue

            # Safety check: se la patch è ancora troppo grande, scaliamo conservativamente
            if w_target > W or h_target > H:
                scale_factor = min(W / w_target, H / h_target) * 0.8
                w_target = max(1, round(w_target * scale_factor))
                h_target = max(1, round(h_target * scale_factor))
                patch_img = cv2.resize(patch_img, (w_target, h_target), interpolation=cv2.INTER_LINEAR)
                patch_mask = cv2.resize(
                    patch_mask.astype(np.uint8), (w_target, h_target),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            
            
            top= random.randint(0, H-h_target)
            left= random.randint(0, W-w_target)

            roi = img_np[top:top + h_target, left:left + w_target]
            mask_bool = patch_mask
            roi[mask_bool] = patch_img[mask_bool]
            img_np[top:top + h_target, left:left + w_target] = roi

            outlier_mask[top:top + h_target, left:left + w_target] |= patch_mask


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
            # clona il target per evitare corruzione database
            target = {k: v.clone() if isinstance(v, torch.Tensor) else v
                      for k, v in target.items()}
            
            H, W = img.shape[-2:]

            if random.random() >= self.paste_prob:
                target["ood_mask"] = torch.zeros(H, W, dtype=torch.bool)
                return img, target

            # Salviamo originali prima del paste per il fallback
            original_img = img
            original_target = {
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in target.items()
            }
            img_pasted, outlier_mask = self.paste_fn(img)
            ood_mask_bool = outlier_mask.bool()

            visible_masks = target["masks"].bool() & ~ood_mask_bool
            valid = visible_masks.flatten(1).any(dim=1)

            if not valid.any():
                # Il paste ha distrutto tutta la supervisione: torniamo all'originale.
                original_target["ood_mask"] = torch.zeros(H, W, dtype=torch.bool)
                return original_img, original_target

            target["masks"] = tv_tensors.Mask(visible_masks[valid])
            target["labels"] = target["labels"][valid]
            target["is_crowd"] = target["is_crowd"][valid]
            target["ood_mask"] = outlier_mask
            img_out = tv_tensors.Image(img_pasted)
            return img_out, target


   
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
