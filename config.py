"""
EXPLICACIÓ SIMPLE: Configuració centralitzada del projecte.

Aquí es defineixen tots els hiperparàmetres (números que controlen com treballa el model).
En lloc de dispersar aquests valors pel codi, es concentren aquí per fàcil modificació.
Canvia els valors aquí si vols ajustar l'entrenament sense tocar el codi principal.

Resultado de esta config (merge polbeltran + uriramos):
  - Backbone ResNet50, layers 3 y 4 descongeladas (fine-tuning de las capas semánticas).
  - Loss Focal + Dice (mejor con clases desbalanceadas como COCO).
  - Augmentaciones moderadas (ver transforms.py).
  - AMP + channels_last + torch.compile + cuDNN benchmark → entrenamiento mucho más rápido.
  - Warmup lineal + cosine annealing (ver main.py).
"""

class Config:
    """Todos los hiperparámetros del modelo y del entrenamiento."""

    # ── Datos ─────────────────────────────────────────────────────────────────
    # COCO por defecto (es lo que pide el enunciado). Para usar VOC2012 (se
    # descarga solo, 21 clases) comenta las 2 líneas de COCO y descomenta las de VOC.
    DATASET     = "COCO"
    NUM_CLASSES = 81          # COCO: 0 = fondo + 80 categorías (category_id remapeados a 1..80)
    # DATASET     = "VOC"
    # NUM_CLASSES = 21        # VOC2012: 20 clases + fondo (+ ignore_index 255 en bordes)

    # Carpeta donde están / se guardan las máscaras pre-generadas de COCO (ver
    # tools/precompute_coco_masks.py). None → usa la misma carpeta de --data-root.
    # Si el COCO es de solo lectura (p.ej. lo descargó el profe), pon aquí una
    # carpeta tuya con permisos de escritura, p.ej.:
    #   MASKS_ROOT = "/home/edxnG05/coco_masks"
    MASKS_ROOT = "/home/edxnG05/coco_masks"

    IMG_SIZE    = 256         # 256 (enunciado). Subir a 384 da más detalle pero ~2x más coste
    BATCH_SIZE  = 32          # con AMP cabe holgado en la L40S 48GB; subir si sobra VRAM
    NUM_WORKERS = 8           # procesos paralelos de carga de datos (~ #cores/3; útil hasta ~12)
    PREFETCH_FACTOR = 4       # batches que precarga cada worker por adelantado

    # ── Optimización de velocidad (solo efecto en GPU; todo opt-out) ──────────
    USE_AMP         = True    # mixed precision fp16 (autocast + GradScaler): ~2x más rápido
    COMPILE         = True    # torch.compile: 1.2-1.8x extra. Ponlo False si da problemas
    CHANNELS_LAST   = True    # memory format channels_last: convoluciones más rápidas
    CUDNN_BENCHMARK = True    # cuDNN elige el algoritmo de conv más rápido (input de tamaño fijo)
    GRAD_CLIP_NORM  = 1.0     # recorte de gradiente (estabilidad con LR altos / capas descongeladas)

    # ── Modelo ────────────────────────────────────────────────────────────────
    BACKBONE   = "resnet152"   # resnet18 | resnet34 | resnet50 | resnet101 | resnet152
    PRETRAINED = True

    # Congelación del encoder capa a capa (True = congelada, False = entrenable):
    #   layer0 → conv inicial + maxpool   (features muy genéricas: bordes)
    #   layer1 → primer bloque ResNet     (features simples: texturas)
    #   layer2 → segundo bloque ResNet    (features medias: formas)
    #   layer3 → tercer bloque ResNet     (features complejas: partes de objetos)  → entrenable
    #   layer4 → cuarto bloque ResNet     (features semánticas: objetos completos) → entrenable
    FREEZE_LAYER0 = True
    FREEZE_LAYER1 = True
    FREEZE_LAYER2 = True
    FREEZE_LAYER3 = True    # descongelada: las capas semánticas necesitan adaptarse a la tarea
    FREEZE_LAYER4 = True     # descongelada: ídem

    # ── Entrenamiento ─────────────────────────────────────────────────────────
    LR_ENCODER   = 1e-5       # bajo: no destruir los pesos ImageNet de las capas descongeladas
    LR_DECODER   = 1e-4       # decoder + head se entrenan desde cero
    WEIGHT_DECAY = 1e-4
    EPOCHS        = 50        # ~30 ya suele bastar con la mejor loss + augmentations + scheduler
    WARMUP_EPOCHS = 2         # warmup lineal del LR antes del cosine annealing

    # Optimizer: "adamw" | "adam" | "sgd" | "rmsprop" | "adagrad"
    OPTIMIZER    = "adamw"
    SGD_MOMENTUM = 0.9        # solo si OPTIMIZER == "sgd"

    # ── Pérdida (Focal + Dice) ────────────────────────────────────────────────
    FOCAL_WEIGHT = 0.5        # peso de la Focal Loss en la combinada
    DICE_WEIGHT  = 0.5        # peso de la Dice Loss
    FOCAL_GAMMA  = 2.0        # exponente de la Focal Loss (enfoca en píxeles difíciles)
    IGNORE_INDEX = 255        # convención VOC para píxeles de borde no etiquetados (inocuo en COCO)

    # ── Reproducibilidad ──────────────────────────────────────────────────────
    SEED = 42
