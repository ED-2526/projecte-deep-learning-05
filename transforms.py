import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms as T


class PairedTransform:
    """
    EXPLICACIÓ SIMPLE: Transformació sincronitzada entre imatge i màscara.
    Fa els mateixos canvis en ambdues (resize, flip, normalització) perquè
    no desincronitzin. La imatge es normalitza (escalat de colors) però
    la màscara es manté com números que representen classes.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)

    def __init__(self, img_size=256, train=True, hflip_p=0.5):
        self.img_size = img_size
        self.train    = train
        self.hflip_p  = hflip_p

    def __call__(self, image, mask):
        """
        EXPLICACIÓ SIMPLE: Aplica les transformacions a la imatge i màscara.
        1. Canvia mida a 256x256 (bilinear per imatge, nearest per màscara)
        2. Flip aleatori (només en entrenament, augmentació de dades)
        3. Normalització ImageNet (només imatge)
        4. Converteix a tensors
        """
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
