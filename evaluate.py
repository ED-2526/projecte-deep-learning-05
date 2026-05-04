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

from classes import get_classes, get_colormap
from config import Config
from engine import validar
from losses import SegmentationLoss
<<<<<<< HEAD
from main import construir_voc
=======
from main import build_dataset
>>>>>>> 7ef54e90e48dd391bd73a11764d25d4ec3dc800e
from metrics import SegmentationMetrics
from models.unet import UNet
from transforms import PairedTransform


<<<<<<< HEAD
def coloritzar_mascara(mask: np.ndarray) -> np.ndarray:
    """
    EXPLICACIÓ SIMPLE: Converteix una màscara en escala de grisos (números) a una imatge colorida RGB.
    Cada número de classe rep un color diferent perquè sigui fàcil de veure visualment.
    Els píxels ignorats es posen en blanc.
    """
=======
def colorize_mask(mask: np.ndarray, colormap: list) -> np.ndarray:
    """Convierte una máscara (H, W) de índices a una imagen RGB (H, W, 3)."""
>>>>>>> 7ef54e90e48dd391bd73a11764d25d4ec3dc800e
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_idx, color in enumerate(colormap):
        rgb[mask == cls_idx] = color
    rgb[mask == 255] = (255, 255, 255)   # ignore_index → blanco
    return rgb


def denormalitzar(image_t: torch.Tensor) -> np.ndarray:
    """
    EXPLICACIÓ SIMPLE: Desfa la normalització que va fer el model.
    Les imatges es normalitzen al principi perquè el model funcioni millor.
    Això les torna a convertir a colors normals que podem veure.
    """
    mean = torch.tensor(PairedTransform.IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(PairedTransform.IMAGENET_STD).view(3, 1, 1)
    img  = image_t.detach().cpu() * std + mean
    img  = img.clamp(0, 1).permute(1, 2, 0).numpy()
<<<<<<< HEAD
    fer_figura(model, dataset, device, num_samples, out_path):
    """
    EXPLICACIÓ SIMPLE: Crea una figura (imatge) que mostra:
    - Primera columna: imatge original
    - Segona columna: màscara correcta (ground truth)
    - Tercera columna: predicció del model
    Això ajuda a veure visualment com de bo és el model.
    """
=======
    return (img * 255).astype(np.uint8)


@torch.no_grad()
def make_figure(model, dataset, device, num_samples, colormap, out_path):
>>>>>>> 7ef54e90e48dd391bd73a11764d25d4ec3dc800e
    fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3 * num_samples))
    if num_samples == 1:
        axes = axes[None, :]

    model.eval()
    for i in range(num_samples):
        image, mask = dataset[i]
        pred = model(image.unsqueeze(0).to(device))[0].argmax(dim=0).cpu().numpy()

        axes[i, 0].imshow(denormalitzar(image))
        axes[i, 0].set_title("input" if i == 0 else "")
        axes[i, 1].imshow(coloritzar_mascara(mask.numpy()))
        axes[i, 1].set_title("ground truth" if i == 0 else "")
        axes[i, 2].imshow(coloritzar_mascaramage))
        axes[i, 0].set_title("input" if i == 0 else "")
        axes[i, 1].imshow(colorize_mask(mask.numpy(), colormap))
        axes[i, 1].set_title("ground truth" if i == 0 else "")
        axes[i, 2].imshow(colorize_mask(pred, colormap))
        axes[i, 2].set_title("prediction" if i == 0 else "")
        for ax in axes[i]:
            ax.axis("off")
carregar_checkpoint(model, ckpt_path):
    """
    EXPLICACIÓ SIMPLE: Carrega els pesos guardats d'un model preentrenat.
    Els pesos són els paràmetres que va aprendre el model durant l'entrenament.
    Això permet usar un model ja entrenat sense haver de tornar a entrenar.
    """
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


def principal(args):
    """
    EXPLICACIÓ SIMPLE: Funció principal per avaluar el model.
    1. Carrega el model entrenat
    2. Carrega les dades de validació
    3. Calcula quina precisió té el model
    4. Mostra les mètriques per classe
    5. Crea una figura amb exemples visuals
    """
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate] dataset = {cfg.DATASET}  |  num_classes = {cfg.NUM_CLASSES}")

<<<<<<< HEAD
    val_ds = construir_voc(args.data_root, "val", cfg.IMG_SIZE)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE,
                            shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True)

    model = UNet(num_classes=num_classes, pretrained=False).to(device)
    carregar_checkpoint(model, args.ckpt)
=======
    val_ds     = build_dataset(args.data_root, "val", cfg)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE,
                            shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True)

    model = UNet(num_classes=cfg.NUM_CLASSES, pretrained=False).to(device)
    load_checkpoint(model, args.ckpt)
>>>>>>> 7ef54e90e48dd391bd73a11764d25d4ec3dc800e

    criterion = SegmentationLoss(cfg.CE_WEIGHT, cfg.DICE_WEIGHT, cfg.IGNORE_INDEX)
<<<<<<< HEAD
    metrics   = SegmentationMetrics(num_classes=num_classes, ignore_index=cfg.IGNORE_INDEX)
    val_loss, val_metrics = validar(model, val_loader, criterion, metrics, device)
=======
    metrics   = SegmentationMetrics(num_classes=cfg.NUM_CLASSES, ignore_index=cfg.IGNORE_INDEX)
    val_loss, val_metrics = validate(model, val_loader, criterion, metrics, device)
>>>>>>> 7ef54e90e48dd391bd73a11764d25d4ec3dc800e

    classes  = get_classes(cfg.DATASET)
    colormap = get_colormap(cfg.DATASET)

    print(f"\n=== {cfg.DATASET} val results ===")
    print(f"val_loss : {val_loss:.4f}")
    print(f"mIoU     : {val_metrics['mIoU']:.4f}")
    print(f"\nIoU per class:")
    iou = val_metrics["IoU_per_class"]
    for name, v in sorted(zip(classes, iou), key=lambda x: x[1]):
        print(f"  {name:<20} {v:.4f}")

    if not args.no_figure:
        out_dir  = Path("docs"); out_dir.mkdir(exist_ok=True)
        out_path = out_dir / "qualitative_results.png"
<<<<<<< HEAD
        fer_figura(model, val_ds, device, args.num_samples, out_path)
=======
        make_figure(model, val_ds, device, args.num_samples, colormap, out_path)
>>>>>>> 7ef54e90e48dd391bd73a11764d25d4ec3dc800e


def analitzar_arguments():
    """
    EXPLICACIÓ SIMPLE: Llegeix els arguments de la línia de comandes per a evaluate.py.
    Els arguments permeten especificar quin checkpoint carregar i com mostrar resultats.
    """
    p = argparse.ArgumentParser(description="Evaluate a trained U-Net checkpoint")
    p.add_argument("--ckpt",        type=str, default="checkpoints/best.pt")
    p.add_argument("--data-root",   type=str, default="./data")
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--no-figure",   action="store_true",
                   help="Skip the qualitative figure (just print metrics)")
    return p.parse_args()


if __name__ == "__main__":
    principal(analitzar_arguments())


if __name__ == "__main__":
    main(parse_args())
