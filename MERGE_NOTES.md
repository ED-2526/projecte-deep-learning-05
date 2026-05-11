# merge_poluri — notas del merge (polbeltran + uriramos)

Esta rama combina las dos ramas de trabajo de velocidad/calidad en una sola, quedándose
con la mejor versión de cada cosa y dejando el código consistente y ejecutable.

## Qué se ha tomado de cada rama

### De `polbeltran` (optimizaciones de velocidad)
- `torch.compile` (con el modelo sin compilar como referencia para el checkpoint; no compila en `--overfit`).
- `channels_last` en el modelo y los inputs.
- `torch.backends.cudnn.benchmark = True`.
- DataLoader con `persistent_workers=True` + `prefetch_factor`.
- Loss calculada en **fp32** bajo AMP (`preds.float()`) — la Dice usa `smooth=1e-6`, que se iría a 0 en fp16.
- `tools/precompute_coco_masks.py` + `CocoSegmentationCached` en `dataset.py` (evitan `pycocotools.annToMask` en cada `__getitem__`).
- `GradScaler` creado **una vez** en `main.py` y pasado al engine (mantiene el factor de escala entre epochs).

### De `uriramos`
- **Augmentaciones** en `transforms.py` (rotación, afí, brillo/contraste, hue/saturación, gamma, blur).
  - *Cambio respecto a su versión:* se ha **quitado el flip vertical** (las imágenes naturales no aparecen del revés → perjudica) y se han **reducido los rangos** de rotación (±15°) y shear (±10°).
- **Focal Loss + Dice** (`losses.py`) en lugar de CE + Dice — mejor con clases desbalanceadas (COCO).
  - *Cambio:* `FocalLoss.alpha = 1.0` (en multiclase el 0.25 de RetinaNet solo escala la loss).
- **Warmup lineal + cosine annealing** (`main.py` / `engine.py`), scheduler avanzado por batch.
- **Gradient clipping** (`Config.GRAD_CLIP_NORM`, por defecto 1.0).
- Descongelar `layer3` y `layer4` del encoder (fine-tuning de las capas semánticas).
- Backbone por defecto `resnet50` (suficiente para COCO, ~4x más rápido que resnet152).
- Scripts auxiliares: `quick_test.py` (reescrito y arreglado), `fast_train.py`, `train_optimized.sh`, `OPTIMIZATION_GUIDE.md`.

## Qué se ha descartado / corregido
- **Marcadores de conflicto sin resolver** (`<<<<<<<` / `=======` / `>>>>>>>`) que estaban commiteados en `config.py` y `main.py` de `uriramos` → resueltos.
- `SUMMARY.txt` y `training_log.txt` de `uriramos` → no incluidos (el primero tenía métricas inventadas/aspiracionales; el segundo era un mensaje de error de una ejecución fallida).
- `LR_DECODER = 1e-2` / `LR_ENCODER = 1e-3` con SGD → demasiado agresivo; se vuelve a `1e-4` / `1e-5` con AdamW (la combinación que dio el mejor resultado del equipo, 0.6241 mIoU), apoyado en el warmup + gradient clipping.
- `IMG_SIZE = 384` y `BATCH_SIZE = 48/64` → `IMG_SIZE = 256` (lo que pide el enunciado) y `BATCH_SIZE = 32` (con holgura en la L40S 48GB). Ambos comentados como alternativas.
- `NUM_CLASSES = 81` → `91` (el dataset COCO asigna `category_id` 1-90 sin remapear, así que hace falta 91).
- `main.py`/`evaluate.py` ahora despachan VOC vs COCO con `construir_dataset(...)` (antes `main.py` solo soportaba VOC aunque el enunciado pide COCO).
- `registre_iou_per_classe` usa `get_classes(cfg.DATASET)` (antes hardcodeaba las 21 clases de VOC → fallaba con COCO).

## Cómo usar
```bash
# sanity check sin dataset (modelo + loss + AMP + métricas)
python quick_test.py

# smoke test rápido con un subset
python fast_train.py --data-root <ruta-dataset> --samples 500 --epochs 3

# pre-generar máscaras de COCO (una sola vez, acelera mucho)
python tools/precompute_coco_masks.py --coco-root <ruta-COCO> --split train
python tools/precompute_coco_masks.py --coco-root <ruta-COCO> --split val

# entrenamiento completo
python main.py --data-root <ruta-dataset> --epochs 30

# evaluación + figura cualitativa
python evaluate.py --ckpt checkpoints/best.pt --data-root <ruta-dataset> --num-samples 8
```

## Pendiente / a vigilar
- La rama `ramacerve` también está integrando COCO; al mergear, conviliar `construir_dataset` con su versión.
- Si AMP da NaNs con `OPTIMIZER="sgd"` y LR altos: bajar `LR_DECODER` o usar `USE_AMP = False`.
- `OPTIMIZATION_GUIDE.md` viene de `uriramos`; algunos números son estimaciones, no medidas.
