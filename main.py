import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from classes import VOC_CLASSES
from config import Config
from engine import train_one_epoch, validate
from losses import SegmentationLoss
from metrics import SegmentationMetrics
from models.unet import UNet
from transforms import PairedTransform


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_voc(root: str, image_set: str, img_size: int):
    transform = PairedTransform(img_size=img_size, train=(image_set == "train"))
    return datasets.VOCSegmentation(
        root=root, year="2012", image_set=image_set, download=True,
        transforms=transform,
    )


def build_optimizer(model: UNet, cfg: Config) -> torch.optim.Optimizer:
    encoder_params = list(model.encoder.parameters())
    decoder_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    return torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": cfg.LR_ENCODER},
            {"params": decoder_params, "lr": cfg.LR_DECODER},
        ],
        weight_decay=cfg.WEIGHT_DECAY,
    )


def per_class_iou_log(iou_per_class, prefix="val_iou"):
    """Construye un dict {val_iou/<class>: iou} para wandb."""
    return {f"{prefix}/{name}": float(iou)
            for name, iou in zip(VOC_CLASSES, iou_per_class)}


def main(args: argparse.Namespace) -> None:
    cfg = Config()
    set_seed(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[main] device = {device}")

    # --- data (VOC2012 baseline) ---
    num_classes = 21
    train_ds = build_voc(args.data_root, "train", cfg.IMG_SIZE)
    val_ds   = build_voc(args.data_root, "val",   cfg.IMG_SIZE)

    if args.overfit > 0:
        idx = list(range(args.overfit))
        train_ds = Subset(train_ds, idx)
        val_ds   = train_ds   # mismas imágenes en val: queremos ver overfit
        print(f"[main] OVERFIT mode on {args.overfit} samples")

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                              shuffle=True,  num_workers=cfg.NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE,
                              shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True)

    # --- model ---
    model = UNet(num_classes=num_classes, pretrained=cfg.PRETRAINED).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[main] U-Net params: {n_params/1e6:.2f}M")

    # --- loss + optimizer + scheduler ---
    criterion = SegmentationLoss(
        ce_weight=cfg.CE_WEIGHT,
        dice_weight=cfg.DICE_WEIGHT,
        ignore_index=cfg.IGNORE_INDEX,
    )
    epochs = args.epochs if args.epochs is not None else cfg.EPOCHS
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    metrics = SegmentationMetrics(num_classes=num_classes, ignore_index=cfg.IGNORE_INDEX)

    # --- wandb ---
    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        wandb.init(
            project="xnap-segmentation",
            name=("overfit" if args.overfit > 0 else "voc-baseline"),
            mode="offline" if args.wandb_offline else "online",
            config={k: getattr(cfg, k) for k in dir(cfg) if k.isupper()},
        )
        wandb.watch(model, criterion, log="all", log_freq=50)

    # --- training loop ---
    ckpt_dir = Path("checkpoints"); ckpt_dir.mkdir(exist_ok=True)
    best_miou = 0.0

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch=epoch)
        val_loss, val_metrics = validate(model, val_loader, criterion, metrics, device, epoch=epoch)
        scheduler.step()

        log = {
            "epoch":       epoch,
            "train_loss":  train_loss,
            "val_loss":    val_loss,
            "val_mIoU":    val_metrics["mIoU"],
            "lr_encoder":  optimizer.param_groups[0]["lr"],
            "lr_decoder":  optimizer.param_groups[1]["lr"],
        }
        log.update(per_class_iou_log(val_metrics["IoU_per_class"]))

        # imprime sólo los escalares principales en consola; el resto va a wandb
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="U-Net segmentation training (VOC2012 baseline)")
    p.add_argument("--data-root", type=str, default="./data",
                   help="Carpeta donde descargar/leer VOC2012")
    p.add_argument("--epochs", type=int, default=None,
                   help="Sobrescribe Config.EPOCHS")
    p.add_argument("--overfit", type=int, default=0,
                   help="Si >0, entrena/valida sobre las primeras N imágenes (sanity check)")
    p.add_argument("--no-wandb", action="store_true", help="Desactiva Wandb")
    p.add_argument("--wandb-offline", action="store_true", help="Wandb en modo offline")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
