import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms as T


class PairedTransform:
    """Transformación sincronizada imagen-máscara para segmentación.

    - Resize: bilinear en imagen, nearest en máscara (preserva los índices de clase).
    - Random horizontal flip aplicado a la vez a imagen y máscara (sólo en train).
    - Normalización ImageNet sobre la imagen.
    - La máscara se devuelve como LongTensor con los índices de clase intactos.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)

    def __init__(self, img_size=256, train=True, hflip_p=0.5):
        self.img_size = img_size
        self.train    = train
        self.hflip_p  = hflip_p

    def __call__(self, image, mask):
        size = (self.img_size, self.img_size)
        image = TF.resize(image, size, interpolation=T.InterpolationMode.BILINEAR)
        mask  = TF.resize(mask,  size, interpolation=T.InterpolationMode.NEAREST)

        if self.train and random.random() < self.hflip_p:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)

        image = TF.to_tensor(image)
        image = TF.normalize(image, self.IMAGENET_MEAN, self.IMAGENET_STD)

        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        return image, mask
