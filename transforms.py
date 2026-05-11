import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms as T
from PIL import Image, ImageFilter


class PairedTransform:
    """
    EXPLICACIÓ SIMPLE: Transformació sincronitzada imatge-màscara per a segmentació.
    Aplica EXACTAMENT els mateixos canvis geomètrics a la imatge i a la màscara
    (resize, flip, rotació, afí) perquè els píxels segueixin alineats amb les classes.
    Els canvis de color (brillantor, contrast, to...) i el blur només s'apliquen a la imatge.
    La imatge es normalitza amb les estadístiques d'ImageNet; la màscara es manté com
    a LongTensor amb els índexs de classe intactes.

    Augmentaciones de entrenamiento (moderadas — evitan las que perjudican imágenes
    naturales, p.ej. el flip vertical: coches/personas no aparecen del revés):
      - Resize a img_size x img_size  (bilinear imagen / nearest máscara)
      - Random horizontal flip
      - Random rotation  ±15°
      - Random affine    (shear ±10°, scale 0.8–1.2)
      - Random brightness / contrast
      - Random hue / saturation
      - Random gamma
      - Random Gaussian blur
    En validación solo se aplica el resize + normalización.
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

        if self.train:
            # --- transformaciones geométricas (sincronizadas imagen + máscara) ---
            if random.random() < self.hflip_p:
                image = TF.hflip(image)
                mask  = TF.hflip(mask)

            if random.random() < 0.5:
                angle = random.uniform(-15, 15)
                image = TF.rotate(image, angle, interpolation=T.InterpolationMode.BILINEAR)
                mask  = TF.rotate(mask,  angle, interpolation=T.InterpolationMode.NEAREST)

            if random.random() < 0.3:
                shear = (random.uniform(-10, 10), random.uniform(-10, 10))
                scale = random.uniform(0.8, 1.2)
                image = TF.affine(image, angle=0, translate=(0, 0), scale=scale,
                                  shear=shear, interpolation=T.InterpolationMode.BILINEAR)
                mask  = TF.affine(mask,  angle=0, translate=(0, 0), scale=scale,
                                  shear=shear, interpolation=T.InterpolationMode.NEAREST)

            # --- transformaciones de color (solo imagen) ---
            if random.random() < 0.5:
                image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
                image = TF.adjust_contrast(image,   random.uniform(0.8, 1.2))

            if random.random() < 0.5:
                image = TF.adjust_hue(image,        random.uniform(-0.1, 0.1))
                image = TF.adjust_saturation(image, random.uniform(0.8, 1.2))

            if random.random() < 0.25:
                gamma = random.uniform(0.8, 1.2)
                arr   = np.asarray(image, dtype=np.float32) / 255.0
                arr   = np.power(arr, gamma)
                image = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))

            if random.random() < 0.25:
                image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

        image = TF.to_tensor(image)
        image = TF.normalize(image, self.IMAGENET_MEAN, self.IMAGENET_STD)
        mask  = torch.from_numpy(np.array(mask, dtype=np.int64))
        return image, mask
