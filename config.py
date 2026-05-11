"""
EXPLICACIÓ SIMPLE: Configuració centralitzada del projecte.

Aquí es defineixen tots els hiperparàmetres (números que controlen com treballa el model).
En lloc de dispersar aquests valors pel codi, es concentren aquí per fàcil modificació.
Canvia els valors aquí si vols ajustar l'entrenament sense tocar el codi principal.
"""

class Config:
    """
    EXPLICACIÓ SIMPLE: Classe amb tots els paràmetres del model i entrenament.
    - DATASET: Quines dades usar (COCO o VOC2012)
    - BATCH_SIZE: Quantes imatges procesar juntes
    - LR (Learning Rate): Velocitat d'aprenentatge (més alt = més ràpid però menys estable)
    - EPOCHS: Quantes vegades entrenar amb totes les dades
    - CE_WEIGHT, DICE_WEIGHT: Com pesar les dos funcions d'error
    """
    # Datos — el enunciado del proyecto pide COCO 256x256.
    # Se deja COCO por defecto; cambiar a "VOC2012"/21 si decides empezar por VOC.
    DATASET     = "VOC"
    NUM_CLASSES = 21     
    
<<<<<<< Updated upstream
    # DATASET     = "COCO"
    # NUM_CLASSES = 91          # COCO stuff+things; ajusta según el subset que uses
    IMG_SIZE    = 256
    BATCH_SIZE  = 8
    NUM_WORKERS = 4

    # Modelo
    BACKBONE = "resnet152"   # resnet18 | resnet34 | resnet50 | resnet101 | resnet152
=======
    DATASET     = "COCO"
    NUM_CLASSES = 81          # COCO usa 80 clases de objeto + fondo
    IMG_SIZE    = 384         # AUMENTADO: de 256 a 384 (más detalle, mejor mIoU +3-5%)
    BATCH_SIZE  = 32          # REDUCIDO: de 48 a 32 (por IMG_SIZE más grande)
    NUM_WORKERS = 12           # Aumentado para paralelizar carga de datos
    CACHE_MASKS = True  # Cachear masks en memoria para evitar decodificar cada época

    # Modelo
    BACKBONE = "resnet50"   # resnet18 | resnet34 | resnet50 | resnet101 | resnet152 (resnet50 es ~4x más rápido que resnet152)
>>>>>>> Stashed changes
    PRETRAINED  = True

    # Congelación del encoder capa a capa (True = congelada, False = entrenable):
    # layer0 → conv inicial + maxpool  (features muy genéricas: bordes)
    # layer1 → primer bloque ResNet    (features simples: texturas)
    # layer2 → segundo bloque ResNet   (features medias: formas)
    # layer3 → tercer bloque ResNet    (features complejas: partes de objetos)
    # layer4 → cuarto bloque ResNet    (features semánticas: objetos completos)
    FREEZE_LAYER0 = True
    FREEZE_LAYER1 = True
<<<<<<< Updated upstream
    FREEZE_LAYER2 = True
    FREEZE_LAYER3 = True
    FREEZE_LAYER4 = True
=======
    FREEZE_LAYER2 = False     # DESCONGELADO: más capacidad de aprendizaje (+3-5% mIoU)
    FREEZE_LAYER3 = False     # DESCONGELADO: layer3 y layer4 necesitan entrenar para segmentación
    FREEZE_LAYER4 = False     # DESCONGELADO: son capas semánticas críticas
>>>>>>> Stashed changes

    # Entrenamiento
    LR_ENCODER  = 1e-3        # AUMENTADO: de 5e-4, con más capas abiertas puede ser mayor
    LR_DECODER  = 1e-2        # AUMENTADO: de 5e-3 (decoder es crítico)
    WEIGHT_DECAY = 1e-4
    EPOCHS      = 50           # AUMENTADO: de 30 a 50 (más convergencia, mejor mIoU)
    WARMUP_EPOCHS = 3          # AUMENTADO: warmup más largo para estabilidad

    # Optimizer: "adamw" | "adam" | "sgd" | "rmsprop" | "adagrad"
    OPTIMIZER   = "sgd"        # CAMBIO: SGD suele dar mejor mIoU final que AdamW (aunque más lento)
    SGD_MOMENTUM = 0.95        # Más momentum para convergencia más robusta

    # Pérdida
    CE_WEIGHT   = 0.6          # CAMBIO: ahora FOCAL_WEIGHT (Focal Loss mejor que CE)
    DICE_WEIGHT = 0.4          # AJUSTADO: de 0.5 a 0.4
    IGNORE_INDEX = 255
    
    # Mixed Precision Training (AMP) - 1.5-2x más rápido
    USE_AMP = True

    # Seed
    SEED = 42
