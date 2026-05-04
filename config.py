class Config:
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
