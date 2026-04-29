import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        # preds:   (B, C, H, W) probabilidades tras softmax
        # targets: (B, H, W) índices de clase
        num_classes = preds.shape[1]
        targets_oh = F.one_hot(targets.long(), num_classes).permute(0, 3, 1, 2).float()
        intersection = (preds * targets_oh).sum(dim=(2, 3))
        union        = preds.sum(dim=(2, 3)) + targets_oh.sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class SegmentationLoss(nn.Module):
    def __init__(self, ce_weight=0.5, dice_weight=0.5, ignore_index=255):
        super().__init__()
        self.ce   = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss()
        self.w_ce, self.w_dice = ce_weight, dice_weight

    def forward(self, preds, targets):
        return self.w_ce   * self.ce(preds, targets.long()) + \
               self.w_dice * self.dice(preds.softmax(dim=1), targets)
