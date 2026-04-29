import torch


class SegmentationMetrics:
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.confusion_matrix = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.long
        )

    def update(self, preds, targets):
        # preds:   (B, C, H, W)
        # targets: (B, H, W)
        preds_flat   = preds.argmax(dim=1).view(-1).cpu()
        targets_flat = targets.view(-1).cpu().long()

        # filtrar el índice a ignorar (ej. 255 en VOC)
        valid = (targets_flat >= 0) & (targets_flat < self.num_classes) & \
                (targets_flat != self.ignore_index)
        preds_flat   = preds_flat[valid]
        targets_flat = targets_flat[valid]

        idx = self.num_classes * targets_flat + preds_flat
        cm  = torch.bincount(idx, minlength=self.num_classes ** 2)
        self.confusion_matrix += cm.reshape(self.num_classes, self.num_classes)

    def compute(self):
        cm   = self.confusion_matrix.float()
        diag = cm.diag()
        iou  = diag / (cm.sum(1) + cm.sum(0) - diag + 1e-6)
        return {
            "mIoU": iou.mean().item(),
            "IoU_per_class": iou.tolist(),
        }
