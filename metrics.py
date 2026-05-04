import torch


class SegmentationMetrics:
    """
    EXPLICACIÓ SIMPLE: Classe que calcula les mètriques de precisió del model.
    Acumula una matriu de confusió (compara prediccions vs veritat) i calcula mIoU.
    mIoU (mean Intersection over Union) és la mètrica principal en segmentació.
    """
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.reinicialitzar()

    def reinicialitzar(self):
        """Neteja la matriu de confusió per começar a comptar de zero."""
        self.confusion_matrix = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.long
        )

    def actualitzar(self, preds, targets):
        """
        Afegeix les prediccions d'aquest batch a la matriu de confusió.
        Compara cada píxel predicció amb la veritat.
        """
        # preds:   (B, C, H, W)
        # targets: (B, H, W)
        preds_flat   = preds.argmax(dim=1).view(-1).cpu()
        targets_flat = targets.view(-1).cpu().long()

        # filtrar el índice a ignorar (ej. 255 en VOC)
        valid = (targets_flat >= 0) & (targets_flat < self.num_classes) & \
                (targets_flat != self.ignore_index)
        preds_flat   = preds_flat[valid]
        targets_flat = targets_flat[valid]
        
        # Debug: mostrar información de validación
        total_pixels = targets.numel()
        valid_pixels = valid.sum().item()
        if valid_pixels == 0:
            print(f"[WARNING] No valid pixels found! Total: {total_pixels}, Valid: {valid_pixels}")
            print(f"[DEBUG] Target values: min={targets_flat.min().item() if len(targets_flat) > 0 else 'N/A'}, max={targets.max().item()}, contains 255: {(targets == 255).sum().item()}")
            return

        idx = self.num_classes * targets_flat + preds_flat
        cm  = torch.bincount(idx, minlength=self.num_classes ** 2)
        self.confusion_matrix += cm.reshape(self.num_classes, self.num_classes)

    def calcular(self):
        """
        Calcula les mètriques finals basades en la matriu de confusió acumulada.
        Retorna mIoU (puntuació global) i IoU per a cada classe.
        """
        cm   = self.confusion_matrix.float()
        total_pixels = cm.sum().item()
        
        if total_pixels == 0:
            print("[WARNING] Confusion matrix is empty! No valid predictions.")
            return {
                "mIoU": 0.0,
                "IoU_per_class": [0.0] * self.num_classes,
            }
        
        diag = cm.diag()
        iou  = diag / (cm.sum(1) + cm.sum(0) - diag + 1e-6)
        
        # Only average over classes present in the ground truth
        classes_present = cm.sum(1) > 0
        if classes_present.sum() > 0:
            miou = iou[classes_present].mean().item()
        else:
            miou = 0.0
            
        return {
            "mIoU": miou,
            "IoU_per_class": iou.tolist(),
        }
