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
    
    # DATASET     = "COCO"
    # NUM_CLASSES = 91          # COCO stuff+things; ajusta según el subset que uses
    IMG_SIZE    = 256
    BATCH_SIZE  = 8
    NUM_WORKERS = 4

    # Modelo
    BACKBONE    = "resnet50"
    PRETRAINED  = True

    # Entrenamiento
    LR_ENCODER  = 1e-5        # bajo: no destruir pesos ImageNet
    LR_DECODER  = 1e-4        # decoder + head se entrenan desde cero
    WEIGHT_DECAY = 1e-4
    EPOCHS      = 50
    OPTIMIZER   = "adamw"

    # Pérdida
    CE_WEIGHT   = 0.5
    DICE_WEIGHT = 0.5
    IGNORE_INDEX = 255

    # Seed
    SEED = 42
