import os
from glob import glob

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_paths  = sorted(glob(os.path.join(img_dir,  "*.jpg")))
        self.mask_paths = sorted(glob(os.path.join(mask_dir, "*.png")))
        assert len(self.img_paths) == len(self.mask_paths), \
            f"#imgs={len(self.img_paths)} != #masks={len(self.mask_paths)}"
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert("RGB")
        mask  = Image.open(self.mask_paths[idx])
        if self.transform:
            image, mask = self.transform(image, mask)
        return image, mask


class CocoSegmentation(Dataset):
    """Segmentación semántica sobre COCO derivada de anotaciones de instancias.

    Cada píxel recibe el category_id de COCO (1-90); el fondo queda en 0.
    Las instancias se superponen de mayor a menor área para que las pequeñas
    no queden tapadas.

    Estructura esperada en <root>:
        <root>/train2017/   <root>/val2017/
        <root>/annotations/instances_train2017.json
        <root>/annotations/instances_val2017.json
    """

    _SPLIT_DIR = {"train": "train2017", "val": "val2017"}

    def __init__(self, root: str, split: str = "train", transforms=None):
        from pycocotools.coco import COCO

        if split not in self._SPLIT_DIR:
            raise ValueError(f"split debe ser 'train' o 'val', no {split!r}")

        split_dir = self._SPLIT_DIR[split]
        ann_file  = os.path.join(root, "annotations", f"instances_{split_dir}.json")
        self.coco     = COCO(ann_file)
        self.img_dir  = os.path.join(root, split_dir)
        self.ids      = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id   = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        image    = Image.open(os.path.join(self.img_dir, img_info["file_name"])).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns    = self.coco.loadAnns(ann_ids)

        h, w = img_info["height"], img_info["width"]
        mask = np.zeros((h, w), dtype=np.uint8)
        # orden descendente de área: las instancias pequeñas se pintan encima
        for ann in sorted(anns, key=lambda a: a["area"], reverse=True):
            m = self.coco.annToMask(ann)
            mask[m > 0] = ann["category_id"]  # IDs 1-90; 0 = fondo

        if self.transforms:
            image, mask = self.transforms(image, Image.fromarray(mask))
        return image, mask
