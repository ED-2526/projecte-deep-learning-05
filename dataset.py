import os
from glob import glob
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
