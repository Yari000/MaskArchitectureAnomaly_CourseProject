## BOZZA PASTER PATCH COCO SU IMMAGINI PER OUTLIER EXPOSURE

# imports...


class CityscapesWithCOCOPaste(Dataset):
    def __init__(self, cityscapes_dataset, coco_root, coco_ann_file, 
                 paste_prob=0.5, num_patches=(1,3)):  # probability of attaching a patch // number of patches
        self.base = cityscapes_dataset
        self.coco = COCO(coco_ann_file)
        self.coco_root = coco_root
        self.paste_prob = paste_prob
        self.num_patches = num_patches
        
        # Categorie COCO da escludere (overlap con dati Cityscapes)
        CITYSCAPES_OVERLAP = {
            'person', 'car', 'truck', 'bus', 'motorcycle', 
            'bicycle', 'traffic light', 'stop sign'
        }
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.valid_cat_ids = [
            c['id'] for c in cats 
            if c['name'] not in CITYSCAPES_OVERLAP
        ]
        # Pre-carica lista di (img_id, ann_id) validi
        self.valid_anns = self._build_ann_list()
    
    def _build_ann_list(self):
        anns = []
        for cat_id in self.valid_cat_ids:
            ann_ids = self.coco.getAnnIds(catIds=[cat_id], iscrowd=False)
            for aid in ann_ids:
                ann = self.coco.loadAnns([aid])[0]
                if ann['area'] > 1000:  # filtra patch troppo piccole per evitare casi troppo particolari
                    anns.append(aid)
        return anns
    
    def _get_random_patch(self):
        """Ritorna (patch_rgb, patch_mask) ritagliati dal bbox dell'annotazione."""
        ann = self.coco.loadAnns([random.choice(self.valid_anns)])[0]
        img_info = self.coco.loadImgs([ann['image_id']])[0]
        
        img_path = os.path.join(self.coco_root, img_info['file_name'])
        img = np.array(Image.open(img_path).convert('RGB'))
        
        # Maschera binaria dall'annotazione COCO
        mask = self.coco.annToMask(ann)  # HxW binaria
        
        # Ritaglia sul bbox
        x, y, w, h = [int(v) for v in ann['bbox']]
        patch_img = img[y:y+h, x:x+w]
        patch_mask = mask[y:y+h, x:x+w]
        
        return patch_img, patch_mask
    
    def __getitem__(self, idx):
        image, label = self.base[idx]  # già preprocessati
        outlier_mask = torch.zeros_like(label, dtype=torch.bool)
        
        if random.random() < self.paste_prob:
            n = random.randint(*self.num_patches)
            image_np = image.permute(1,2,0).numpy()  # HxWxC
            H, W = image_np.shape[:2]
            
            for _ in range(n):
                patch_img, patch_mask = self._get_random_patch()
                
                # Scala la patch a dimensione ragionevole
                scale = random.uniform(0.05, 0.2)
                new_h = int(H * scale)
                new_w = int(patch_img.shape[1] * new_h / patch_img.shape[0])
                patch_img = cv2.resize(patch_img, (new_w, new_h))
                patch_mask = cv2.resize(patch_mask, (new_w, new_h), 
                                       interpolation=cv2.INTER_NEAREST)
                
                # Posizione random
                if H - new_h <= 0 or W - new_w <= 0:
                    continue
                top = random.randint(0, H - new_h)
                left = random.randint(0, W - new_w)
                
                # Incolla solo sui pixel della maschera
                roi = image_np[top:top+new_h, left:left+new_w]
                pm = patch_mask.astype(bool)
                roi[pm] = patch_img[pm]
                image_np[top:top+new_h, left:left+new_w] = roi
                
                # Aggiorna outlier mask
                outlier_mask[top:top+new_h, left:left+new_w] |= torch.from_numpy(pm)
            
            image = torch.from_numpy(image_np).permute(2,0,1)
        
        return image, label, outlier_mask
    
    def __len__(self):
        return len(self.base)
