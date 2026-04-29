"""Evaluación final + visualización cualitativa.

Carga un checkpoint, evalúa el mIoU sobre el split de validación y guarda una
figura con N ejemplos (imagen | ground truth | predicción).

Uso:
    python evaluate.py --ckpt checkpoints/best.pt --num-samples 8
    python evaluate.py --ckpt checkpoints/best.pt --data-root ./data --no-figure
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from classes import VOC_CLASSES, VOC_COLORMAP
from config import Config
from engine import validate
from losses import SegmentationLoss
from main import build_voc
from metrics import SegmentationMetrics
from models.unet import UNet
from transforms import PairedTransform


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convierte una máscara (H, W) de índices a una imagen RGB (H, W, 3)."""
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_idx, color in enumerate(VOC_COLORMAP):
        rgb[mask == cls_idx] = color
    rgb[mask == 255] = (255, 255, 255)   # ignore_index → blanco
    return rgb


def denormalize(image_t: torch.Tensor) -> np.ndarray:
    """Deshace la normalización ImageNet y devuelve un array (H, W, 3) uint8."""
    mean = torch.tensor(PairedTransform.IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(PairedTransform.IMAGENET_STD).view(3, 1, 1)
    img  = image_t.detach().cpu() * std + mean
    img  = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


@torch.no_grad()
def make_figure(model, dataset, device, num_samples, out_path):
    fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3 * num_samples))
    if num_samples == 1:
        axes = axes[None, :]

    model.eval()
    for i in range(num_samples):
        image, mask = dataset[i]
        pred = model(image.unsqueeze(0).to(device))[0].argmax(dim=0).cpu().numpy()

        axes[i, 0].imshow(denormalize(image))
        axes[i, 0].set_title("input" if i == 0 else "")
        axes[i, 1].imshow(colorize_mask(mask.numpy()))
        axes[i, 1].set_title("ground truth" if i == 0 else "")
        axes[i, 2].imshow(colorize_mask(pred))
        axes[i, 2].set_title("prediction" if i == 0 else "")
        for ax in axes[i]:
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate] figure saved to {out_path}")


def load_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    if "mIoU" in ckpt:
        print(f"[evaluate] loaded checkpoint @ epoch {ckpt.get('epoch', '?')} (mIoU={ckpt['mIoU']:.4f})")


def main(args):
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 21

    val_ds = build_voc(args.data_root, "val", cfg.IMG_SIZE)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE,
                            shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True)

    model = UNet(num_classes=num_classes, pretrained=False).to(device)
    load_checkpoint(model, args.ckpt)

    # --- métricas cuantitativas ---
    criterion = SegmentationLoss(cfg.CE_WEIGHT, cfg.DICE_WEIGHT, cfg.IGNORE_INDEX)
    metrics   = SegmentationMetrics(num_classes=num_classes, ignore_index=cfg.IGNORE_INDEX)
    val_loss, val_metrics = validate(model, val_loader, criterion, metrics, device)

    print(f"\n=== VOC2012 val results ===")
    print(f"val_loss : {val_loss:.4f}")
    print(f"mIoU     : {val_metrics['mIoU']:.4f}")
    print(f"\nIoU per class:")
    iou = val_metrics["IoU_per_class"]
    for name, v in sorted(zip(VOC_CLASSES, iou), key=lambda x: x[1]):
        print(f"  {name:<14} {v:.4f}")

    # --- figura cualitativa ---
    if not args.no_figure:
        out_dir = Path("docs"); out_dir.mkdir(exist_ok=True)
        out_path = out_dir / "qualitative_results.png"
        make_figure(model, val_ds, device, args.num_samples, out_path)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained U-Net checkpoint")
    p.add_argument("--ckpt", type=str, default="checkpoints/best.pt")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--no-figure", action="store_true",
                   help="Skip the qualitative figure (just print metrics)")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
