import os
os.environ.setdefault("TORCH_HOME", r"C:\torch_cache")

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from classes import get_classes
from config import Config
from dataset import CocoSegmentation, CocoSegmentationCached
from engine import entrenar_una_epoca, validar
from losses import SegmentationLoss
from metrics import SegmentationMetrics
from models.unet import UNet
from transforms import PairedTransform


def establir_llavor(seed: int) -> None:
    """Fija las semillas aleatorias para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def construir_dataset(cfg: Config, root: str, split: str):
    """Construye el dataset (train/val) según cfg.DATASET.

    - VOC : torchvision.datasets.VOCSegmentation (se descarga solo).
    - COCO: CocoSegmentationCached si hay máscaras pre-generadas
            (ver tools/precompute_coco_masks.py); si no, CocoSegmentation
            (genera las máscaras al vuelo — mucho más lento).
    """
    transform = PairedTransform(img_size=cfg.IMG_SIZE, train=(split == "train"))
    dataset = cfg.DATASET.upper()

    if dataset in ("VOC", "VOC2012"):
        image_set = "train" if split == "train" else "val"
        return datasets.VOCSegmentation(root=root, year="2012", image_set=image_set,
                                        download=True, transforms=transform)

    if dataset == "COCO":
        masks_root = getattr(cfg, "MASKS_ROOT", None) or root
        cached_dir = os.path.join(masks_root, f"masks_{'train2017' if split == 'train' else 'val2017'}")
        if os.path.isdir(cached_dir):
            return CocoSegmentationCached(root=root, split=split, transforms=transform,
                                          masks_root=masks_root)
        print(f"[main] Aviso: no hay máscaras pre-generadas en {cached_dir}; se usa "
              f"CocoSegmentation (lento). Genera las máscaras con tools/precompute_coco_masks.py "
              f"(usa --masks-root si el COCO es de solo lectura).")
        return CocoSegmentation(root=root, split=split, transforms=transform)

    raise ValueError(f"DATASET desconocido: {cfg.DATASET!r}. Usa 'VOC' o 'COCO'.")


def construir_optimitzador(model: UNet, cfg: Config) -> torch.optim.Optimizer:
    """Crea el optimizer con learning rates separados para encoder (bajo) y decoder (alto)."""
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    decoder_params = [p for n, p in model.named_parameters()
                      if not n.startswith("encoder.") and p.requires_grad]
    param_groups = []
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": cfg.LR_ENCODER})
    param_groups.append({"params": decoder_params, "lr": cfg.LR_DECODER})

    name = cfg.OPTIMIZER.lower()
    if name == "adamw":
        return torch.optim.AdamW(param_groups, weight_decay=cfg.WEIGHT_DECAY)
    if name == "adam":
        return torch.optim.Adam(param_groups, weight_decay=cfg.WEIGHT_DECAY)
    if name == "sgd":
        return torch.optim.SGD(param_groups, momentum=cfg.SGD_MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
    if name == "rmsprop":
        return torch.optim.RMSprop(param_groups, weight_decay=cfg.WEIGHT_DECAY)
    if name == "adagrad":
        return torch.optim.Adagrad(param_groups, weight_decay=cfg.WEIGHT_DECAY)
    raise ValueError(f"Optimizer desconocido: {cfg.OPTIMIZER!r}. "
                     f"Usa: adamw | adam | sgd | rmsprop | adagrad")


def build_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps):
    """LambdaLR: warmup lineal hasta warmup_steps, luego cosine annealing hasta total_steps."""
    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def registre_iou_per_classe(iou_per_classe, class_names, prefix="val_iou"):
    """Dict {val_iou/<clase>: iou} para Wandb (omite las clases 'N/A' de COCO)."""
    return {f"{prefix}/{name}": float(iou)
            for name, iou in zip(class_names, iou_per_classe)
            if name != "N/A"}


def principal(args: argparse.Namespace) -> None:
    cfg = Config()
    establir_llavor(cfg.SEED)
    if getattr(cfg, "CUDNN_BENCHMARK", False):
        torch.backends.cudnn.benchmark = True   # tamaño de input fijo → cuDNN puede autotunear
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[main] device = {device}  |  dataset = {cfg.DATASET}  |  num_classes = {cfg.NUM_CLASSES}")

    # ── datos ────────────────────────────────────────────────────────────────
    train_ds = construir_dataset(cfg, args.data_root, "train")
    val_ds   = construir_dataset(cfg, args.data_root, "val")

    if args.overfit > 0:
        idx = list(range(args.overfit))
        train_ds = Subset(train_ds, idx)
        val_ds   = train_ds   # mismas imágenes en val: queremos ver overfit
        print(f"[main] OVERFIT mode on {args.overfit} samples")

    loader_kwargs = dict(num_workers=cfg.NUM_WORKERS, pin_memory=True)
    if cfg.NUM_WORKERS > 0:
        loader_kwargs.update(persistent_workers=True,
                             prefetch_factor=getattr(cfg, "PREFETCH_FACTOR", 2))
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False, **loader_kwargs)

    # ── modelo ───────────────────────────────────────────────────────────────
    model = UNet(num_classes=cfg.NUM_CLASSES, backbone=cfg.BACKBONE, pretrained=cfg.PRETRAINED).to(device)

    freeze_map = {
        "layer0": cfg.FREEZE_LAYER0, "layer1": cfg.FREEZE_LAYER1, "layer2": cfg.FREEZE_LAYER2,
        "layer3": cfg.FREEZE_LAYER3, "layer4": cfg.FREEZE_LAYER4,
    }
    frozen = []
    for layer_name, should_freeze in freeze_map.items():
        if should_freeze:
            for param in getattr(model.encoder, layer_name).parameters():
                param.requires_grad = False
            frozen.append(layer_name)
    if frozen:
        print(f"[main] Capas congeladas: {', '.join(frozen)}")

    n_params    = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[main] U-Net params: {n_params/1e6:.2f}M total | {n_trainable/1e6:.2f}M entrenables")

    # ── optimizaciones de velocidad (solo en GPU) ────────────────────────────
    use_amp        = getattr(cfg, "USE_AMP", False) and device.type == "cuda"
    channels_last  = getattr(cfg, "CHANNELS_LAST", False) and device.type == "cuda"
    grad_clip_norm = getattr(cfg, "GRAD_CLIP_NORM", 0.0)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
        print("[main] channels_last activado")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        print("[main] AMP (mixed precision fp16) activado")

    # torch.compile acelera forward/backward; mantenemos `model` sin compilar como
    # referencia para guardar el checkpoint (evita el prefijo '_orig_mod.' en el state_dict).
    # No se compila en modo --overfit: la compilación inicial no compensa con pocos pasos.
    train_model = model
    if getattr(cfg, "COMPILE", False) and device.type == "cuda" and args.overfit == 0:
        try:
            train_model = torch.compile(model)
            print("[main] torch.compile activado")
        except Exception as e:
            print(f"[main] torch.compile no disponible ({e}); se usa el modelo sin compilar")

    # ── loss + optimizer + scheduler + métricas ──────────────────────────────
    criterion = SegmentationLoss(focal_weight=cfg.FOCAL_WEIGHT, dice_weight=cfg.DICE_WEIGHT,
                                 gamma=getattr(cfg, "FOCAL_GAMMA", 2.0),
                                 ignore_index=cfg.IGNORE_INDEX)
    epochs    = args.epochs if args.epochs is not None else cfg.EPOCHS
    optimizer = construir_optimitzador(model, cfg)

    steps_per_epoch = max(1, len(train_loader))
    warmup_steps    = getattr(cfg, "WARMUP_EPOCHS", 0) * steps_per_epoch
    total_steps     = epochs * steps_per_epoch
    scheduler       = build_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)

    metrics     = SegmentationMetrics(num_classes=cfg.NUM_CLASSES, ignore_index=cfg.IGNORE_INDEX)
    class_names = get_classes(cfg.DATASET)

    # ── wandb ────────────────────────────────────────────────────────────────
    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        frozen_layers = [f"L{i}" for i, f in enumerate(freeze_map.values()) if f]
        freeze_str = f"freeze({'_'.join(frozen_layers)})" if frozen_layers else "nofrozen"
        run_name = "overfit" if args.overfit > 0 else \
                   f"{cfg.DATASET.lower()}-{cfg.BACKBONE}-{cfg.OPTIMIZER}-{freeze_str}"
        wandb.init(project="finetuning", name=run_name,
                   mode="offline" if args.wandb_offline else "online",
                   config={k: getattr(cfg, k) for k in dir(cfg) if k.isupper()})
        wandb.watch(model, criterion, log="all", log_freq=50)

    # ── bucle de entrenamiento ───────────────────────────────────────────────
    ckpt_dir = Path("checkpoints"); ckpt_dir.mkdir(exist_ok=True)
    best_miou = 0.0

    for epoch in range(epochs):
        train_loss = entrenar_una_epoca(train_model, train_loader, optimizer, criterion, device,
                                        scaler=scaler, scheduler=scheduler, use_amp=use_amp,
                                        channels_last=channels_last, grad_clip_norm=grad_clip_norm,
                                        epoch=epoch)
        val_loss, val_metrics = validar(train_model, val_loader, criterion, metrics, device,
                                        use_amp=use_amp, channels_last=channels_last, epoch=epoch)

        log = {
            "epoch":       epoch,
            "train_loss":  train_loss,
            "val_loss":    val_loss,
            "val_mIoU":    val_metrics["mIoU"],
            "lr_encoder":  optimizer.param_groups[0]["lr"] if len(optimizer.param_groups) > 1 else 0.0,
            "lr_decoder":  optimizer.param_groups[-1]["lr"],
        }
        log.update(registre_iou_per_classe(val_metrics["IoU_per_class"], class_names))

        main_keys = ("epoch", "train_loss", "val_loss", "val_mIoU")
        line = " | ".join(f"{k}={log[k]:.4f}" if isinstance(log[k], float) else f"{k}={log[k]}"
                          for k in main_keys)
        print(f"[epoch {epoch:03d}] {line}")

        if use_wandb:
            wandb.log(log)

        if val_metrics["mIoU"] > best_miou:
            best_miou = val_metrics["mIoU"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),   # `model` sin compilar
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
    p = argparse.ArgumentParser(description="U-Net segmentation training (VOC2012 / COCO)")
    p.add_argument("--data-root", type=str, default="./data",
                   help="Carpeta donde descargar/leer el dataset (raíz de COCO si DATASET='COCO')")
    p.add_argument("--epochs", type=int, default=None, help="Sobrescribe Config.EPOCHS")
    p.add_argument("--overfit", type=int, default=0,
                   help="Si >0, entrena/valida sobre las primeras N imágenes (sanity check)")
    p.add_argument("--no-wandb", action="store_true", help="Desactiva Wandb")
    p.add_argument("--wandb-offline", action="store_true", help="Wandb en modo offline")
    return p.parse_args()


if __name__ == "__main__":
    principal(analitzar_arguments())
