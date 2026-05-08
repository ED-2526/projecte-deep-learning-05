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
    #DATASET     = "VOC"
    #NUM_CLASSES = 21     
    
    DATASET     = "COCO"
    NUM_CLASSES = 80          # COCO stuff+things; ajusta según el subset que uses
    IMG_SIZE    = 256
    BATCH_SIZE  = 32
    NUM_WORKERS = 0  # 0 en CPU (multiprocessing overhead es más caro que el beneficio)
    CACHE_MASKS = True  # Cachear masks en memoria para evitar decodificar cada época

    # Modelo
    BACKBONE = "resnet50"   # resnet18 | resnet34 | resnet50 | resnet101 | resnet152 (resnet50 es ~4x más rápido que resnet152)
    PRETRAINED  = True

    # Congelación del encoder capa a capa (True = congelada, False = entrenable):
    # layer0 → conv inicial + maxpool  (features muy genéricas: bordes)
    # layer1 → primer bloque ResNet    (features simples: texturas)
    # layer2 → segundo bloque ResNet   (features medias: formas)
    # layer3 → tercer bloque ResNet    (features complejas: partes de objetos)
    # layer4 → cuarto bloque ResNet    (features semánticas: objetos completos)
    FREEZE_LAYER0 = True
    FREEZE_LAYER1 = True
    FREEZE_LAYER2 = False  # Descongelar para aprender mejor con COCO
    FREEZE_LAYER3 = False
    FREEZE_LAYER4 = False

    # Entrenamiento
    LR_ENCODER  = 1e-4        # bajo: no destruir pesos ImageNet
    LR_DECODER  = 1e-4        # decoder + head se entrenan desde cero
    WEIGHT_DECAY = 1e-4
    EPOCHS      = 50

    # Optimizer: "adamw" | "adam" | "sgd" | "rmsprop" | "adagrad"
    OPTIMIZER   = "adamw"
    SGD_MOMENTUM = 0.9        # solo usado si OPTIMIZER = "sgd"

    # Pérdida
    CE_WEIGHT   = 0.5
    DICE_WEIGHT = 0.5
    IGNORE_INDEX = 255

    # Seed
    SEED = 42
