import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss: Enfatiza los píxeles difíciles de clasificar.
    Especialmente útil para clases desbalanceadas (como en COCO).
    gamma=2 es un buen valor por defecto.
    """
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, preds, targets):
        # preds: (B, C, H, W)
        # targets: (B, H, W)
        num_classes = preds.shape[1]
        targets = targets.long()
        
        # Crear máscara para píxeles a ignorar
        valid_mask = targets != self.ignore_index
        targets_safe = targets.clone()
        targets_safe[~valid_mask] = 0
        
        # Calcular CE
        ce_loss = F.cross_entropy(preds, targets_safe, reduction='none')
        
        # Aplicar máscara
        ce_loss = ce_loss * valid_mask.float()
        
        # Calcular probabilidades
        p = torch.exp(-ce_loss)
        
        # Focal loss = -alpha * (1-p)^gamma * ce_loss
        focal_weight = self.alpha * torch.pow(1 - p, self.gamma)
        focal_loss = focal_weight * ce_loss
        
        return focal_loss.mean()


class DiceLoss(nn.Module):
    """
    EXPLICACIÓ SIMPLE: Calcula la Dice Loss, una mètrica que mesura la superposició
    entre la predicció i la veritat. Útil quan les classes són desbalanceades 
    (una classe domina molt més que les altres).
    """
    def __init__(self, smooth=1e-6, ignore_index=255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, preds, targets):
        # preds:   (B, C, H, W) probabilidades tras softmax
        # targets: (B, H, W) índices de clase
        num_classes = preds.shape[1]
        targets = targets.long()
        
        # Crear máscara para píxeles a ignorar
        valid_mask = targets != self.ignore_index
        
        # Reemplazar valores de ignore_index con 0 para evitar error en one_hot
        targets_safe = targets.clone()
        targets_safe[~valid_mask] = 0
        
        targets_oh = F.one_hot(targets_safe, num_classes).permute(0, 3, 1, 2).float()
        
        # Aplicar máscara de validez a los targets one-hot
        targets_oh = targets_oh * valid_mask.unsqueeze(1).float()
        
        intersection = (preds * targets_oh).sum(dim=(2, 3))
        union        = preds.sum(dim=(2, 3)) + targets_oh.sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class SegmentationLoss(nn.Module):
    """
    EXPLICACIÓ SIMPLE: Combina dos tipus d'error per entrenar el model:
    1. Focal Loss: Mejor que CE para clases desbalanceadas (como COCO)
    2. Dice Loss: Error especialitzat per desbalanceo de classes
    Els dos es pesen 0.5 cada un. Usar ambdós fa que el model aprengui millor
    les classes que són rares o petites.
    """
    def __init__(self, focal_weight=0.6, dice_weight=0.4, ignore_index=255):
        super().__init__()
        self.focal = FocalLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)
        self.w_focal, self.w_dice = focal_weight, dice_weight

    def forward(self, preds, targets):
        return self.w_focal * self.focal(preds, targets) + \
               self.w_dice * self.dice(preds.softmax(dim=1), targets)
