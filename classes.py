"""Nombres de clase y paletas de color para los datasets soportados."""

# --- VOC2012 ---
VOC_CLASSES = (
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)
assert len(VOC_CLASSES) == 21

VOC_COLORMAP = [
    (0, 0, 0),       (128, 0, 0),     (0, 128, 0),    (128, 128, 0),
    (0, 0, 128),     (128, 0, 128),   (0, 128, 128),  (128, 128, 128),
    (64, 0, 0),      (192, 0, 0),     (64, 128, 0),   (192, 128, 0),
    (64, 0, 128),    (192, 0, 128),   (64, 128, 128), (192, 128, 128),
    (0, 64, 0),      (128, 64, 0),    (0, 192, 0),    (128, 192, 0),
    (0, 64, 128),
]
assert len(VOC_COLORMAP) == 21


# --- COCO (índice 0 = fondo, índices 1-90 = category_id de COCO) ---
# Los IDs de COCO no son contiguos; las posiciones sin categoría se marcan con "N/A".
COCO_CLASSES = (
    "background",
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella",
    "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "N/A", "dining table", "N/A", "N/A",
    "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "N/A", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
)
assert len(COCO_CLASSES) == 91


def _gen_colormap(n: int):
    """Genera n colores distintos usando el mismo algoritmo de bits que VOC."""
    def bit(val, idx):
        return (val >> idx) & 1

    colormap = []
    for i in range(n):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= bit(c, 0) << (7 - j)
            g |= bit(c, 1) << (7 - j)
            b |= bit(c, 2) << (7 - j)
            c >>= 3
        colormap.append((r, g, b))
    return colormap


COCO_COLORMAP = _gen_colormap(91)


def get_classes(dataset: str) -> tuple:
    d = dataset.upper()
    if d in ("VOC", "VOC2012"):
        return VOC_CLASSES
    if d == "COCO":
        return COCO_CLASSES
    raise ValueError(f"Dataset desconocido: {dataset!r}. Usa 'VOC' o 'COCO'.")


def get_colormap(dataset: str) -> list:
    d = dataset.upper()
    if d in ("VOC", "VOC2012"):
        return VOC_COLORMAP
    if d == "COCO":
        return COCO_COLORMAP
    raise ValueError(f"Dataset desconocido: {dataset!r}. Usa 'VOC' o 'COCO'.")
