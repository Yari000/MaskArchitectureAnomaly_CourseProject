import torch
import torch.nn.functional as F
from torch.optim import AdamW

from training.mask_classification_semantic import MaskClassificationSemantic


def rba_loss(
    per_pixel_scores: torch.Tensor,
    ood_mask: torch.Tensor,
    alpha: float = 5.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Implementa la RBA loss (del paper) sui pixel OOD

    -per_pixel_scores: tensore [B, C, H, W] con score semantici per classe
    -ood_mask: tensore [B, H, W] con T su pixel anomali
    -alpha: hinge margin

    """

    ood_mask = ood_mask.to(per_pixel_scores.device).bool()
    if not ood_mask.any():
        # Se non ci sono pixel ood la loss è nulla
        return per_pixel_scores.new_zeros(())

    # Tanh comprime i valori e rende il processo numericamente stabile
    score = torch.tanh(per_pixel_scores)
    rba = -score.sum(dim=1)
    # hinge loss quadratica che penalizza i pixel sotto-margine
    loss_map = F.relu(alpha - rba).pow(2)
    selected = loss_map[ood_mask]

    if reduction == "sum":
        return selected.sum()
    if reduction == "mean":
        print("mean is currently used")
        return selected.mean()
    raise ValueError(f"Riduzione ignota: {reduction}")


class MaskClassificationSemanticOE(MaskClassificationSemantic):
    """
    Estensione di maskclassificationsemantic+
    - lambda_rba: peso della loss 
    - rba_alpha: margine della hinge loss
    - il training_step() restituisce la loss totale
    - configure_optimizers() crea AdamW optimizer su parametri trainable
    """

    def __init__(
        self,
        *args,
        lambda_rba: float = 0.1,
        rba_alpha: float = 5.0,
        rba_reduction: str = "mean",
        freeze_heads_only: bool = True,
        **kwargs,
    ) -> None:
        # Costruiamo la classe base che contiene tutto
        super().__init__(*args, **kwargs)
        self.lambda_rba = lambda_rba
        self.rba_alpha = rba_alpha
        self.rba_reduction = rba_reduction
        self.freeze_heads_only = freeze_heads_only

        if freeze_heads_only:
            # Applichiamo il freeze da paper (solo mask e class head unfrozen)
            self.freeze_all_but_heads()

    def freeze_all_but_heads(self) -> None:

        for param in self.network.parameters():
            param.requires_grad = False

        for module_name in ("class_head", "mask_head"):
            module = getattr(self.network, module_name)
            for param in module.parameters():
                param.requires_grad = True

        trainable = [name for name, p in self.named_parameters() if p.requires_grad]

    def on_fit_start(self) -> None:

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        self.print(f"Trainable parameters: {trainable:,} / {total:,}")

    def configure_optimizers(self):

        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = AdamW(
            trainable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        return optimizer


    def _extract_ood_masks(
        self,
        targets: list[dict],
        size: tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Estrae e riscala le oodmask, passando altezza e larghezza attese
        
        """

        masks = []
        for target in targets:
            mask = target.get("ood_mask")
            if mask is None:
                mask = torch.zeros(size, dtype=torch.bool)

            mask = mask.to(device)
            if tuple(mask.shape[-2:]) != tuple(size):
                # Rescale per adattare le dimensioni
                mask = F.interpolate( mask[None, None].float(), size=size,
                    mode="nearest", )[0, 0].bool()
            masks.append(mask.bool())

        return torch.stack(masks, dim=0)

    # Analogo del train step in lightning module
    def training_step(self, batch, batch_idx):

        imgs, targets = batch

        # Passo forward EoMT
        mask_logits_per_block, class_logits_per_block = self(imgs)

        losses_all_blocks = {}
        for i, (mask_logits, class_logits) in enumerate(
            zip(mask_logits_per_block, class_logits_per_block)
        ):
            losses = self.criterion(
                masks_queries_logits=mask_logits,
                class_queries_logits=class_logits,
                targets=targets,
            )
            block_postfix = self.block_postfix(i)
            losses_all_blocks |= {
                f"{key}{block_postfix}": value for key, value in losses.items()
            }

        # loss standard di segmmentation
        seg_loss = self.criterion.loss_total(losses_all_blocks, self.log)

        # Usiamo l'ultimo blocco per rba loss
        final_mask_logits = F.interpolate(
            mask_logits_per_block[-1],
            size=imgs.shape[-2:],
            mode="bilinear",
        )
        per_pixel_scores = self.to_per_pixel_logits_semantic(
            final_mask_logits,
            class_logits_per_block[-1],
        )
        ood_masks = self._extract_ood_masks(
            targets=targets,
            size=imgs.shape[-2:],
            device=imgs.device,
        )
        rba_loss_val = rba_loss(
            per_pixel_scores=per_pixel_scores,
            ood_mask=ood_masks,
            alpha=self.rba_alpha,
            reduction=self.rba_reduction,
        )

        # la loss finale è la somma pesata 
        total_loss = seg_loss + self.lambda_rba * rba_loss_val

        # log Wandb
        self.log(
            "losses_epoch/train_loss_total_oe",
            total_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )
        self.log(
            "losses_epoch/train_rba_loss",
            rba_loss_val,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "losses_epoch/train_loss_without_rba",
            seg_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        with torch.no_grad():
             if ood_masks.any():
                rba_scores = -torch.tanh(per_pixel_scores).sum(dim=1)
                self.log("rba_diag/ood_mean", rba_scores[ood_masks].mean())
                self.log("rba_diag/id_mean",  rba_scores[~ood_masks].mean())
                self.log("rba_diag/id_ood_gap",
                     rba_scores[~ood_masks].mean() - rba_scores[ood_masks].mean())
                active = (self.rba_alpha - rba_scores[ood_masks] > 0).float().mean()
                self.log("rba_diag/hinge_active_fraction", active)
               
        return total_loss
