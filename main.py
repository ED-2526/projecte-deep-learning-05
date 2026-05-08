import os
os.environ.setdefault("TORCH_HOME", r"C:\torch_cache")

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from PIL import Image
import pycocotools.mask as maskUtils

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


class COCOSegmentationWrapper(Dataset):
    """
    WRAPPER OPTIMIZADO para COCO. Converteix annotations RLE a masks PIL.
    Con cache=True, precachea todas las masks al init (más rápido durante training).
    """
    def __init__(self, coco_dataset, transforms=None, cache_masks=False):
        self.coco_dataset = coco_dataset
        self.transforms = transforms
        self._mask_cache = {} if cache_masks else None
        
    def __len__(self):
        return len(self.coco_dataset)
    
    def __getitem__(self, idx):
        image, target = self.coco_dataset[idx]
        
        # Usar cache si está disponible
        if self._mask_cache is not None:
            if idx not in self._mask_cache:
                self._mask_cache[idx] = self._build_mask(image, target)
            mask = self._mask_cache[idx]
        else:
            mask = self._build_mask(image, target)
        
        # Aplicar transformaciones
        if self.transforms:
            image, mask = self.transforms(image, mask)
        
        return image, mask
    
    def _build_mask(self, image, target):
        """Build segmentation mask from COCO RLE annotations."""
        if not target:
            H, W = image.size[1], image.size[0]
            return Image.fromarray(np.zeros((H, W), dtype=np.uint8))
        
        H, W = image.size[1], image.size[0]
        mask = np.zeros((H, W), dtype=np.uint8)
        
        for ann in target:
            if "segmentation" not in ann:
                continue
            segmentation = ann["segmentation"]
            if not isinstance(segmentation, dict):  # Solo RLE
                continue
            
            try:
                category_id = min(ann.get("category_id", 0), 255)  # Cap at 255 for uint8
                rle_mask = maskUtils.decode(segmentation)
                mask[rle_mask > 0] = category_id
            except Exception:
                pass
        
        return Image.fromarray(mask)



def construir_coco(root: str, image_set: str, img_size: int, cache_masks: bool = False):
    """
    EXPLICACIÓ SIMPLE: Carrega el dataset COCO (imatges i anotacions).
    COCO2017 té train2017, val2017, i test2017.
    Aplica transformacions per adaptar les imatges al tamaño correcte i fer augmentació.
    """
    transform = PairedTransform(img_size=img_size, train=(image_set == "train"))
    
    # Map image_set to COCO year folder
    year_map = {"train": "train2017", "val": "val2017"}
    coco_year = year_map.get(image_set, "val2017")
    
    # Carrega COCO sense transformacions (les aplicarem al wrapper)
    coco_dataset = datasets.CocoDetection(
        root=f"{root}/{coco_year}",
        annFile=f"{root}/annotations/instances_{coco_year}.json",
        transforms=None,  # No transforms aquí; les aplicarem al wrapper
    )
    
    # Wrap per convertir annotations a segmentation masks
    wrapper = COCOSegmentationWrapper(coco_dataset, transforms=transform, cache_masks=cache_masks)
    
    # Pre-cache all masks if requested (para val, no es mucho: 5k imágenes)
    if cache_masks:
        print(f"[construir_coco] Pre-caching masks para {image_set} ({len(wrapper)} imágenes)...")
        for i in range(len(wrapper)):
            if i % max(1, len(wrapper)//10) == 0:
                print(f"  {i}/{len(wrapper)}")
            _ = wrapper[i]  # Trigger caching
        print(f"[construir_coco] Masks cacheadas!")
    
    return wrapper


def construir_optimitzador(model: UNet, cfg: Config) -> torch.optim.Optimizer:
    """
    EXPLICACIÓ SIMPLE: Crea l'optimitzador que actualitza els pesos del model.
    Usa velocitats d'aprenentatge (learning rates) diferents per a l'encoder i el decoder:
    - Encoder: velocitat baixa (no trencar els pesos preentrenats)
    - Decoder: velocitat alta (aprendre des de zero)
    L'optimitzador es tria segons cfg.OPTIMIZER: adamw | adam | sgd | rmsprop | adagrad
    """
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    decoder_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
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
    raise ValueError(f"Optimizer desconocido: {cfg.OPTIMIZER!r}. Usa: adamw | adam | sgd | rmsprop | adagrad")


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

    # --- data (COCO2017) ---
    num_classes = 80  # COCO has 80 object classes
    # Cache val masks (5k imágenes = ~1.5GB RAM pero vale la pena)
    # Train masks: demasiadas (118k), solo usar transforms
    train_ds = construir_coco(args.data_root, "train", cfg.IMG_SIZE, cache_masks=False)
    val_ds   = construir_coco(args.data_root, "val",   cfg.IMG_SIZE, cache_masks=True)

    if args.overfit > 0:
        idx = list(range(args.overfit))
        train_ds = Subset(train_ds, idx)
        val_ds   = train_ds
        print(f"[main] OVERFIT mode on {args.overfit} samples")

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                              shuffle=True,  num_workers=cfg.NUM_WORKERS, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE,
                              shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=False)

    model = UNet(num_classes=cfg.NUM_CLASSES, backbone=cfg.BACKBONE, pretrained=cfg.PRETRAINED).to(device)

    freeze_map = {
        "layer0": cfg.FREEZE_LAYER0,
        "layer1": cfg.FREEZE_LAYER1,
        "layer2": cfg.FREEZE_LAYER2,
        "layer3": cfg.FREEZE_LAYER3,
        "layer4": cfg.FREEZE_LAYER4,
    }
    frozen = []
    for layer_name, should_freeze in freeze_map.items():
        if should_freeze:
            for param in getattr(model.encoder, layer_name).parameters():
                param.requires_grad = False
            frozen.append(layer_name)
    if frozen:
        print(f"[main] Capes congelades: {', '.join(frozen)}")

    n_params    = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[main] U-Net params: {n_params/1e6:.2f}M total | {n_trainable/1e6:.2f}M entrenables")

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
        frozen_layers = [f"L{i}" for i, f in enumerate([
            cfg.FREEZE_LAYER0, cfg.FREEZE_LAYER1, cfg.FREEZE_LAYER2,
            cfg.FREEZE_LAYER3, cfg.FREEZE_LAYER4
        ]) if f]
        freeze_str = f"freeze({'_'.join(frozen_layers)})" if frozen_layers else "nofrozen"
        run_name = f"coco-{cfg.BATCH_SIZE}-{cfg.BACKBONE}"
        wandb.init(
            project="COCO",
            entity="1671799-uab",
            group="batch",
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
            "lr_encoder":  optimizer.param_groups[0]["lr"] if len(optimizer.param_groups) > 1 else 0.0,
            "lr_decoder":  optimizer.param_groups[-1]["lr"],
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
    p = argparse.ArgumentParser(description="U-Net segmentation training (COCO2017)")
    p.add_argument("--data-root", type=str, default="/home/datasets/coco",
                   help="Carpeta donde leer el dataset COCO")
    p.add_argument("--epochs", type=int, default=None,
                   help="Sobrescribe Config.EPOCHS")
    p.add_argument("--overfit", type=int, default=0,
                   help="Si >0, entrena/valida sobre las primeras N imágenes (sanity check)")
    p.add_argument("--no-wandb", action="store_true", help="Desactiva Wandb")
    p.add_argument("--wandb-offline", action="store_true", help="Wandb en modo offline")
    return p.parse_args()


if __name__ == "__main__":
    principal(analitzar_arguments())
