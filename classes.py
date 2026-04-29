"""Nombres de clase para los datasets soportados."""

VOC_CLASSES = (
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)
assert len(VOC_CLASSES) == 21


# Paleta VOC oficial (en RGB) — útil para colorear máscaras al visualizar.
VOC_COLORMAP = [
    (0, 0, 0),       (128, 0, 0),     (0, 128, 0),    (128, 128, 0),
    (0, 0, 128),     (128, 0, 128),   (0, 128, 128),  (128, 128, 128),
    (64, 0, 0),      (192, 0, 0),     (64, 128, 0),   (192, 128, 0),
    (64, 0, 128),    (192, 0, 128),   (64, 128, 128), (192, 128, 128),
    (0, 64, 0),      (128, 64, 0),    (0, 192, 0),    (128, 192, 0),
    (0, 64, 128),
]
assert len(VOC_COLORMAP) == 21
