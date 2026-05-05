import os
os.environ.setdefault("TORCH_HOME", r"C:\torch_cache")

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from classes import VOC_CLASSES
from config import Config
from engine import entrenar_una_epoca, validar
from losses import SegmentationLoss
from metrics import SegmentationMetrics
from models.unet import UNet
from transforms import PairedTransform
from torchvision import datasets


def establir_llavor(seed: int) -> None:
    """
    EXPLICACIÓ SIMPLE: Fixa el número aleatori inicial (llavor) per a la reproducibilitat.
    Fent que tots els algoritmes aleatoris generin la mateixa sequència cada vegada.
    Útil per a que els resultats siguin iguals cada vegada que executem el codi.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def construir_voc(root: str, image_set: str, img_size: int):
    """
    EXPLICACIÓ SIMPLE: Carrega el dataset VOC (imatges i màscares).
    Si no existeix, el descarrega automàticament. Aplica transformacions per adaptar
    les imatges al tamaño correcte i fer augmentació de dades si és entrenament.
    """
    transform = PairedTransform(img_size=img_size, train=(image_set == "train"))
    return datasets.VOCSegmentation(
        root=root, year="2012", image_set=image_set, download=True,
        transforms=transform,
    )


def construir_optimitzador(model: UNet, cfg: Config) -> torch.optim.Optimizer:
    """
    EXPLICACIÓ SIMPLE: Crea l'optimitzador que actualitza els pesos del model.
    Usa velocitats d'aprenentatge (learning rates) diferents per a l'encoder i el decoder:
    - Encoder: velocitat baixa (no trencar els pesos preentrenats)
    - Decoder: velocitat alta (aprendre des de zero)
    """
    encoder_params = list(model.encoder.parameters())
    decoder_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    return torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": cfg.LR_ENCODER},
            {"params": decoder_params, "lr": cfg.LR_DECODER},
        ],
        weight_decay=cfg.WEIGHT_DECAY,
    )


def registre_iou_per_classe(iou_per_classe, prefix="val_iou"):
    """
    EXPLICACIÓ SIMPLE: Crea un diccionari amb les puntuacions d'IoU (precisió) per a cada classe.
    Útil per a veure quines classes el model aprèn bé i quines no.
    El diccionari es veu bé en el sistema de logging Wandb.
    """
    return {f"{prefix}/{name}": float(iou)
            for name, iou in zip(VOC_CLASSES, iou_per_classe)}


def principal(args: argparse.Namespace) -> None:
    """
    EXPLICACIÓ SIMPLE: Aquesta és la funció principal. Organitza tot el procés:
    1. Carrega les dades
    2. Crea el model
    3. Entrena el model durant varis epochs
    4. Guarda el millor model
    5. Mostra resultats i gràfiques
    """
    cfg = Config()
    establir_llavor(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[main] device = {device}  |  dataset = {cfg.DATASET}  |  num_classes = {cfg.NUM_CLASSES}")

    # --- data (VOC2012 baseline) ---
    num_classes = 21
    train_ds = construir_voc(args.data_root, "train", cfg.IMG_SIZE)
    val_ds   = construir_voc(args.data_root, "val",   cfg.IMG_SIZE)

    if args.overfit > 0:
        idx = list(range(args.overfit))
        train_ds = Subset(train_ds, idx)
        val_ds   = train_ds
        print(f"[main] OVERFIT mode on {args.overfit} samples")

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                              shuffle=True,  num_workers=cfg.NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE,
                              shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True)

    model = UNet(num_classes=cfg.NUM_CLASSES, pretrained=cfg.PRETRAINED).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[main] U-Net params: {n_params/1e6:.2f}M")

    criterion = SegmentationLoss(
        ce_weight=cfg.CE_WEIGHT,
        dice_weight=cfg.DICE_WEIGHT,
        ignore_index=cfg.IGNORE_INDEX,
    )
    epochs = args.epochs if args.epochs is not None else cfg.EPOCHS
    optimizer = construir_optimitzador(model, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    metrics   = SegmentationMetrics(num_classes=cfg.NUM_CLASSES, ignore_index=cfg.IGNORE_INDEX)

    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        run_name = "overfit" if args.overfit > 0 else f"{cfg.DATASET.lower()}-baseline"
        wandb.init(
            project="xnap-segmentation",
            name=run_name,
            mode="offline" if args.wandb_offline else "online",
            config={k: getattr(cfg, k) for k in dir(cfg) if k.isupper()},
        )
        wandb.watch(model, criterion, log="all", log_freq=50)

    ckpt_dir = Path("checkpoints"); ckpt_dir.mkdir(exist_ok=True)
    best_miou = 0.0

    for epoch in range(epochs):
        train_loss = entrenar_una_epoca(model, train_loader, optimizer, criterion, device, epoch=epoch)
        val_loss, val_metrics = validar(model, val_loader, criterion, metrics, device, epoch=epoch)
        scheduler.step()

        log = {
            "epoch":       epoch,
            "train_loss":  train_loss,
            "val_loss":    val_loss,
            "val_mIoU":    val_metrics["mIoU"],
            "lr_encoder":  optimizer.param_groups[0]["lr"],
            "lr_decoder":  optimizer.param_groups[1]["lr"],
        }
        log.update(registre_iou_per_classe(val_metrics["IoU_per_class"]))

        main_keys = ("epoch", "train_loss", "val_loss", "val_mIoU")
        line = " | ".join(
            f"{k}={log[k]:.4f}" if isinstance(log[k], float) else f"{k}={log[k]}"
            for k in main_keys
        )
        print(f"[epoch {epoch:03d}] {line}")

        if use_wandb:
            wandb.log(log)

        if val_metrics["mIoU"] > best_miou:
            best_miou = val_metrics["mIoU"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "mIoU": best_miou,
                    "config": {k: getattr(cfg, k) for k in dir(cfg) if k.isupper()},
                },
                ckpt_dir / "best.pt",
            )
            print(f"[epoch {epoch:03d}] new best mIoU = {best_miou:.4f} → checkpoint guardado")

    print(f"[main] Done. Best mIoU = {best_miou:.4f}")
    if use_wandb:
        wandb.summary["best_mIoU"] = best_miou
        wandb.finish()


def analitzar_arguments() -> argparse.Namespace:
    """
    EXPLICACIÓ SIMPLE: Llegeix els arguments de la línia de comandes.
    Els arguments permeten controlar com s'executa el programa:
    - data-root: on guardar/llegir les dades
    - epochs: quants epochs entrenar
    - overfit: mode de prova (entrenar amb pocs datos)
    - wandb: activar o desactivar el logging
    """
    p = argparse.ArgumentParser(description="U-Net segmentation training (VOC2012 baseline)")
    p.add_argument("--data-root", type=str, default="./data",
                   help="Carpeta donde descargar/leer el dataset")
    p.add_argument("--epochs", type=int, default=None,
                   help="Sobrescribe Config.EPOCHS")
    p.add_argument("--overfit", type=int, default=0,
                   help="Si >0, entrena/valida sobre las primeras N imágenes (sanity check)")
    p.add_argument("--no-wandb", action="store_true", help="Desactiva Wandb")
    p.add_argument("--wandb-offline", action="store_true", help="Wandb en modo offline")
    return p.parse_args()


if __name__ == "__main__":
    principal(analitzar_arguments())
