import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms as T
from PIL import Image, ImageFilter


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
        Augmentaciones AGRESIVAS entrenament:
        - Resize a IMG_SIZE (384x384 per més detall)
        - Random horizontal/vertical flip
        - Random rotation ±20 grados (més agresiu)
        - Random affine (shear, scale més agresiu)
        - Random brightness/contrast/hue/saturation
        - Random GaussianBlur
        """
        size = (self.img_size, self.img_size)
        image = TF.resize(image, size, interpolation=T.InterpolationMode.BILINEAR)
        mask  = TF.resize(mask,  size, interpolation=T.InterpolationMode.NEAREST)

        if self.train:
            # Random horizontal flip
            if random.random() < self.hflip_p:
                image = TF.hflip(image)
                mask  = TF.hflip(mask)
            
            # Random vertical flip (40% - més agresiu)
            if random.random() < 0.4:
                image = TF.vflip(image)
                mask  = TF.vflip(mask)
            
            # Random rotation (-20 a +20 grados - més agresiu)
            if random.random() < 0.6:
                angle = random.uniform(-20, 20)
                image = TF.rotate(image, angle, interpolation=T.InterpolationMode.BILINEAR)
                mask  = TF.rotate(mask, angle, interpolation=T.InterpolationMode.NEAREST)
            
            # Random affine (shear -20 a 20, scale 0.7 a 1.3 - més agresiu)
            if random.random() < 0.5:
                shear = (random.uniform(-20, 20), random.uniform(-20, 20))
                scale = random.uniform(0.7, 1.3)
                image = TF.affine(image, angle=0, translate=(0, 0), scale=scale, 
                                shear=shear, interpolation=T.InterpolationMode.BILINEAR)
                mask  = TF.affine(mask, angle=0, translate=(0, 0), scale=scale, 
                                shear=shear, interpolation=T.InterpolationMode.NEAREST)
            
            # Random brightness/contrast (rang més ampli)
            if random.random() < 0.6:
                brightness_factor = random.uniform(0.7, 1.3)
                contrast_factor = random.uniform(0.7, 1.3)
                image = TF.adjust_brightness(image, brightness_factor)
                image = TF.adjust_contrast(image, contrast_factor)
            
            # Random hue/saturation
            if random.random() < 0.5:
                hue = random.uniform(-0.15, 0.15)
                saturation = random.uniform(0.7, 1.3)
                image = TF.adjust_hue(image, hue)
                image = TF.adjust_saturation(image, saturation)
            
            # Random gamma adjustment
            if random.random() < 0.3:
                gamma = random.uniform(0.8, 1.2)
                img_array = np.array(image, dtype=np.float32) / 255.0
                img_array = np.power(img_array, gamma)
                img_array = (img_array * 255).astype(np.uint8)
                image = Image.fromarray(img_array)
            
            # Random Gaussian Blur
            if random.random() < 0.3:
                radius = random.uniform(0.5, 1.5)
                image = image.filter(ImageFilter.GaussianBlur(radius=radius))

        image = TF.to_tensor(image)
        image = TF.normalize(image, self.IMAGENET_MEAN, self.IMAGENET_STD)

        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        return image, mask
