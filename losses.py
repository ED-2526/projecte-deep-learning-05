import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    EXPLICACIÓ SIMPLE: Focal Loss — Cross-Entropy que pesa más los píxeles difíciles.
    El factor (1 - p)^gamma reduce el peso de los píxeles que el modelo ya acierta
    con confianza, así que se centra en los difíciles. Va bien con clases muy
    desbalanceadas (mucho fondo, pocos objetos pequeños), como en COCO.

    gamma=0  → es exactamente Cross-Entropy.
    gamma=2  → valor estándar (RetinaNet).
    """
    def __init__(self, gamma=2.0, alpha=1.0, ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, preds, targets):
        # preds:   (B, C, H, W) logits
        # targets: (B, H, W) índices de clase (o ignore_index)
        targets = targets.long()
        ce = F.cross_entropy(preds, targets, ignore_index=self.ignore_index, reduction="none")
        pt = torch.exp(-ce)                       # p_t = prob. de la clase correcta
        focal = self.alpha * (1.0 - pt) ** self.gamma * ce
        # F.cross_entropy ya pone 0 en los píxeles ignorados; promediamos sobre los válidos
        valid = (targets != self.ignore_index)
        n = valid.sum().clamp(min=1)
        return focal.sum() / n


class DiceLoss(nn.Module):
    """
    EXPLICACIÓ SIMPLE: Dice Loss — mide la superposición entre la predicción y el
    ground truth, normalizada por el tamaño de ambos. Da el mismo peso relativo a
    objetos pequeños y grandes → compensa el desbalanceo de clases.
    """
    def __init__(self, smooth=1e-6, ignore_index=255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, preds, targets):
        # preds:   (B, C, H, W) probabilidades (tras softmax)
        # targets: (B, H, W) índices de clase
        num_classes = preds.shape[1]
        targets = targets.long()

        valid_mask   = targets != self.ignore_index
        targets_safe = targets.clone()
        targets_safe[~valid_mask] = 0             # one_hot peta con valores fuera de rango

        targets_oh = F.one_hot(targets_safe, num_classes).permute(0, 3, 1, 2).float()
        targets_oh = targets_oh * valid_mask.unsqueeze(1).float()   # anula los píxeles ignorados

        intersection = (preds * targets_oh).sum(dim=(2, 3))
        union        = preds.sum(dim=(2, 3)) + targets_oh.sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class SegmentationLoss(nn.Module):
    """
    EXPLICACIÓ SIMPLE: Pérdida combinada para segmentación.
      loss = focal_weight · FocalLoss + dice_weight · DiceLoss
    - FocalLoss: clasificación píxel a píxel, enfocada en los píxeles difíciles.
    - DiceLoss : maximiza el solapamiento por clase, compensa el desbalanceo.
    Las dos se complementan: la primera da gradientes estables, la segunda evita
    que el modelo abandone las clases minoritarias.

    NOTA: a la FocalLoss se le pasan los logits crudos (F.cross_entropy aplica
    log_softmax internamente); a la DiceLoss, las probabilidades (softmax).
    """
    def __init__(self, focal_weight=0.5, dice_weight=0.5, gamma=2.0, ignore_index=255):
        super().__init__()
        self.focal = FocalLoss(gamma=gamma, ignore_index=ignore_index)
        self.dice  = DiceLoss(ignore_index=ignore_index)
        self.w_focal, self.w_dice = focal_weight, dice_weight

    def forward(self, preds, targets):
        targets = targets.long()
        return self.w_focal * self.focal(preds, targets) + \
               self.w_dice  * self.dice(preds.softmax(dim=1), targets)
