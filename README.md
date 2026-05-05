# Segmentación Semántica con Imágenes Naturales — XNAP Project 05

Proyecto del curso *Xarxes Neuronals i Aprenentatge Profund* (Grau d'Enginyeria de Dades, UAB, 2026) — segmentación semántica píxel a píxel sobre imágenes naturales usando una **U-Net con encoder ResNet50 preentrenado en ImageNet**.

> **Objetivo**: dada una imagen RGB, predecir un mapa `(H × W)` donde cada píxel tiene asignada una clase semántica.

---

## 1. Tarea y datos

- **Tarea**: segmentación semántica multiclase (una clase por píxel).
- **Dataset principal**: [COCO](https://cocodataset.org), redimensionado a **256 × 256**.
- **Dataset inicial (baseline rápido)**: [PASCAL VOC 2012 Segmentation](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/), 21 clases (incluye fondo + `ignore_index=255`). Se descarga directamente con `torchvision.datasets.VOCSegmentation`.

Empezamos por VOC porque su tamaño es manejable (~11K imágenes), está soportado nativamente por `torchvision` y permite tener un baseline funcional rápido antes de migrar a COCO.

---

## 2. Arquitectura

**U-Net** con skip connections en 4 escalas. Encoder ResNet50 preentrenado, decoder con `ConvTranspose2d` + concatenación + 2× `Conv-BN-ReLU`, head `Conv 1×1` que mapea a `num_classes`. El forward final aplica una interpolación bilinear para devolver el output al tamaño exacto de la entrada.

```
Input (B, 3, 256, 256)
    │
    ▼
[Encoder ResNet50 preentrenado]
    │  layer0 (1/4)   ───────┐
    │  layer1 (1/4)   ─────┐ │
    │  layer2 (1/8)   ───┐ │ │
    │  layer3 (1/16)  ─┐ │ │ │
    ▼                  │ │ │ │
[Bottleneck layer4 (1/32, 2048ch)]
    │                  │ │ │ │
    ▼                  │ │ │ │
[Decoder: 4 × (Up + skip + 2× Conv-BN-ReLU)]
    │  ◄───────────────┘ │ │ │
    │  ◄─────────────────┘ │ │
    │  ◄───────────────────┘ │
    │  ◄─────────────────────┘
    ▼
[Conv 1×1 → num_classes]  →  bilinear → (B, num_classes, 256, 256)
```

---

## 3. Pérdida y métrica

- **Pérdida combinada `0.5·CE + 0.5·Dice`**
  - Cross-Entropy con `ignore_index=255` — bien condicionada y robusta.
  - Dice Loss — compensa el desbalanceo de clases (fondo dominante).
- **Métrica principal**: **mIoU** (mean Intersection over Union) calculado sobre la confusion matrix acumulada del epoch completo (no promediando por batch). También se reporta **IoU por clase**.

---

## 4. Estructura del repositorio

```
.
├── README.md            # este archivo
├── LICENSE
├── environment.yml      # entorno conda
├── main.py              # punto de entrada — orquesta todo
├── config.py            # hiperparámetros
├── dataset.py           # SegmentationDataset (modo manual con img_dir + mask_dir)
├── transforms.py        # transform sincronizado imagen-máscara
├── losses.py            # DiceLoss + SegmentationLoss combinada
├── metrics.py           # SegmentationMetrics (confusion matrix → mIoU)
├── engine.py            # train_one_epoch() + validate()
├── models/
│   ├── __init__.py
│   └── unet.py          # Encoder ResNet50 + DecoderBlock + UNet
├── docs/
│   └── informe_seguimiento_1.pdf
└── test/                # checks de GitHub Classroom (no tocar)
```

---

## 5. Instalación

```bash
conda activate grupo-5
```

Para tracking de experimentos (Wandb):

```bash
wandb login
```

---

## 6. Cómo reproducir

### Sanity check — overfit con 5 imágenes
Demuestra que el pipeline aprende:

```bash
python main.py --overfit 5 --epochs 30 --no-wandb
```

La loss debe bajar a casi cero. Si no, hay un bug en el pipeline.

### Entrenamiento completo en VOC2012
La primera vez descarga el dataset (~2 GB):

```bash
python main.py --data-root ./data --epochs 50
```

### Argumentos disponibles
| Flag | Descripción |
|---|---|
| `--data-root` | Carpeta donde descargar/leer VOC. Por defecto `./VOCDataset` |
| `--epochs` | Sobrescribe `Config.EPOCHS` |
| `--overfit N` | Entrena/valida sobre las primeras `N` imágenes (sanity check) |
| `--no-wandb` | Desactiva el logging a Wandb |
| `--wandb-offline` | Ejecuta Wandb en modo offline |

Los checkpoints se guardan en `checkpoints/best.pt` (mejor mIoU de validación).

---

## 7. Decisiones de diseño

| Decisión | Razón |
|---|---|
| ResNet50 como encoder (no VGG) | Sin FC al final, skip connections residuales internas, mejor flujo de gradiente |
| Pesos ImageNet | Transfer learning aprovecha features de bajo/medio nivel |
| `LR_ENCODER` (1e-5) ≪ `LR_DECODER` (1e-4) | Encoder preentrenado: bajar LR evita destruir pesos. Decoder se entrena desde cero |
| `ConvTranspose2d` | Upsampling con parámetros aprendibles |
| CE + Dice | CE penaliza correctamente; Dice compensa desbalanceo |
| `IMG_SIZE = 256` | Lo pide el enunciado del proyecto |
| `ignore_index = 255` | Convención de VOC para píxeles de borde no etiquetados |

---

## 8. Estado actual

- [x] Dataset (`SegmentationDataset`) y transform sincronizado
- [x] U-Net con ResNet50 preentrenado
- [x] Loss combinada (CE + Dice)
- [x] Métricas (mIoU + IoU por clase)
- [x] Training loop + validación
- [x] `main.py` con CLI (overfit, wandb, epochs)
- [x] Wandb integrado
- [ ] Sanity check de overfit ejecutado
- [ ] Entrenamiento completo en VOC
- [ ] Migración a COCO

---

## 9. Contributors

Grup 05 — *Xarxes Neuronals i Aprenentatge Profund*, Grau d'Enginyeria de Dades, UAB, 2026.

| Nom | Email UAB |
|---|---|
| (a completar) | |
| (a completar) | |
| (a completar) | |

---

## 10. Referencias

- Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015.
- He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016.
- COCO dataset: <https://cocodataset.org>
- PASCAL VOC 2012: <http://host.robots.ox.ac.uk/pascal/VOC/voc2012/>
