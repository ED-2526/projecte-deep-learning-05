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
      - Random scale (0.5x – 2.0x) + random crop a img_size x img_size
        (sustituye al resize fijo: introduce variación de escala y de posición)
      - Random horizontal flip
      - Random rotation  ±15°  (fill=255 en la máscara → ignore_index)
      - Random affine shear ±10°  (fill=255 en la máscara → ignore_index)
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
        if self.train:
            # --- Random scale (0.5x – 2.0x) + random crop a img_size x img_size ---
            # Sustituye al resize fijo en entrenamiento para introducir variación de escala
            # y de posición del objeto. La máscara se rellena con 255 (ignore_index) en
            # las zonas sin contenido para que la loss no las cuente.
            scale_factor = random.uniform(0.5, 2.0)
            w, h = image.size
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            image = TF.resize(image, (new_h, new_w), interpolation=T.InterpolationMode.BILINEAR)
            mask  = TF.resize(mask,  (new_h, new_w), interpolation=T.InterpolationMode.NEAREST)

            pad_h = max(self.img_size - new_h, 0)
            pad_w = max(self.img_size - new_w, 0)
            if pad_h > 0 or pad_w > 0:
                image = TF.pad(image, [0, 0, pad_w, pad_h], fill=0)
                mask  = TF.pad(mask,  [0, 0, pad_w, pad_h], fill=255)

            cur_w, cur_h = image.size
            top  = random.randint(0, cur_h - self.img_size)
            left = random.randint(0, cur_w - self.img_size)
            image = TF.crop(image, top, left, self.img_size, self.img_size)
            mask  = TF.crop(mask,  top, left, self.img_size, self.img_size)

            # --- transformaciones geométricas (sincronizadas imagen + máscara) ---
            if random.random() < self.hflip_p:
                image = TF.hflip(image)
                mask  = TF.hflip(mask)

            if random.random() < 0.5:
                angle = random.uniform(-15, 15)
                image = TF.rotate(image, angle, interpolation=T.InterpolationMode.BILINEAR)
                mask  = TF.rotate(mask,  angle, interpolation=T.InterpolationMode.NEAREST, fill=255)

            if random.random() < 0.3:
                shear = (random.uniform(-10, 10), random.uniform(-10, 10))
                image = TF.affine(image, angle=0, translate=(0, 0), scale=1.0,
                                  shear=shear, interpolation=T.InterpolationMode.BILINEAR)
                mask  = TF.affine(mask,  angle=0, translate=(0, 0), scale=1.0,
                                  shear=shear, interpolation=T.InterpolationMode.NEAREST, fill=255)

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
        else:
            # Validación: resize determinista (sin augmentation)
            size  = (self.img_size, self.img_size)
            image = TF.resize(image, size, interpolation=T.InterpolationMode.BILINEAR)
            mask  = TF.resize(mask,  size, interpolation=T.InterpolationMode.NEAREST)

        image = TF.to_tensor(image)
        image = TF.normalize(image, self.IMAGENET_MEAN, self.IMAGENET_STD)
        mask  = torch.from_numpy(np.array(mask, dtype=np.int64))
        return image, mask
