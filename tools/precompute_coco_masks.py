"""Pre-genera las máscaras de segmentación de COCO a disco (PNG).

Convertir polígonos de anotaciones → máscara de píxeles con pycocotools en cada
__getitem__ es el principal cuello de botella al entrenar con COCO (118k imágenes
× 50 epochs). Este script lo hace UNA sola vez y guarda las máscaras como PNG,
para luego cargarlas directamente (mucho más rápido) con CocoSegmentationCached.

Uso:
    python tools/precompute_coco_masks.py --coco-root /ruta/a/COCO --split train
    python tools/precompute_coco_masks.py --coco-root /ruta/a/COCO --split val

Estructura esperada en --coco-root:
    train2017/   val2017/
    annotations/instances_train2017.json
    annotations/instances_val2017.json

Salida:
    <coco-root>/masks_train2017/000000XXXXXX.png   (1 canal; valor = category_id 1-90; 0 = fondo)
    <coco-root>/masks_val2017/...

Después, en dataset.py / main.py usa CocoSegmentationCached(root, split, transforms=PairedTransform(...)).
"""
import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


_SPLIT_DIR = {"train": "train2017", "val": "val2017"}


def precompute(coco_root: str, split: str, overwrite: bool = False) -> None:
    from pycocotools.coco import COCO

    split_dir = _SPLIT_DIR[split]
    ann_file  = os.path.join(coco_root, "annotations", f"instances_{split_dir}.json")
    if not os.path.isfile(ann_file):
        raise FileNotFoundError(f"No existe el fichero de anotaciones: {ann_file}")

    coco    = COCO(ann_file)
    out_dir = Path(coco_root) / f"masks_{split_dir}"
    out_dir.mkdir(parents=True, exist_ok=True)

    img_ids = list(coco.imgs.keys())
    print(f"[precompute] {split}: {len(img_ids)} imágenes → {out_dir}")

    skipped = 0
    for img_id in tqdm(img_ids, desc=f"masks {split}"):
        img_info = coco.loadImgs(img_id)[0]
        stem     = Path(img_info["file_name"]).stem
        out_path = out_dir / f"{stem}.png"
        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        h, w    = img_info["height"], img_info["width"]
        mask    = np.zeros((h, w), dtype=np.uint8)
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns    = coco.loadAnns(ann_ids)
        # de mayor a menor área: las instancias pequeñas se pintan encima
        for ann in sorted(anns, key=lambda a: a["area"], reverse=True):
            m = coco.annToMask(ann)
            mask[m > 0] = ann["category_id"]   # IDs 1-90; 0 = fondo

        Image.fromarray(mask).save(out_path)

    print(f"[precompute] hecho. {skipped} ya existían y se saltaron "
          f"(usa --overwrite para regenerarlas).")


def main() -> None:
    p = argparse.ArgumentParser(description="Pre-genera las máscaras de COCO a disco")
    p.add_argument("--coco-root", required=True, help="Carpeta raíz de COCO")
    p.add_argument("--split", choices=["train", "val"], default="train")
    p.add_argument("--overwrite", action="store_true", help="Regenerar aunque ya existan")
    args = p.parse_args()
    precompute(args.coco_root, args.split, args.overwrite)


if __name__ == "__main__":
    main()
