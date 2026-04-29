"""Genera el PDF del informe de seguimiento #1.

Uso:
    python docs/generate_report.py

Salida:
    docs/informe_seguimiento_1.pdf
"""
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, PageBreak, KeepTogether
)


OUT = Path(__file__).parent / "informe_seguimiento_1.pdf"


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
ss = getSampleStyleSheet()

H1 = ParagraphStyle(
    "H1", parent=ss["Heading1"], fontSize=18, spaceAfter=14,
    textColor=colors.HexColor("#1a1a1a"),
)
H2 = ParagraphStyle(
    "H2", parent=ss["Heading2"], fontSize=13, spaceBefore=14, spaceAfter=8,
    textColor=colors.HexColor("#1a1a1a"),
)
H3 = ParagraphStyle(
    "H3", parent=ss["Heading3"], fontSize=11, spaceBefore=8, spaceAfter=4,
    textColor=colors.HexColor("#333333"),
)
BODY = ParagraphStyle(
    "Body", parent=ss["BodyText"], fontSize=10, leading=14,
    alignment=TA_JUSTIFY, spaceAfter=6,
)
BULLET = ParagraphStyle(
    "Bullet", parent=BODY, leftIndent=14, bulletIndent=4, spaceAfter=2,
    alignment=TA_LEFT,
)
CODE = ParagraphStyle(
    "Code", parent=ss["Code"], fontSize=9, leading=11,
    backColor=colors.HexColor("#f5f5f5"),
    borderColor=colors.HexColor("#dcdcdc"), borderWidth=0.5, borderPadding=6,
    spaceAfter=8,
)
CAPTION = ParagraphStyle(
    "Caption", parent=BODY, fontSize=9, textColor=colors.HexColor("#555555"),
    alignment=TA_LEFT, spaceAfter=10,
)


def P(text, style=BODY):
    return Paragraph(text, style)


def bullets(items):
    return [Paragraph(f"&bull;&nbsp; {it}", BULLET) for it in items]


def two_col_table(rows, col_widths=(5.5 * cm, 11.5 * cm)):
    t = Table(rows, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor("#e8e8e8")),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("BOX",          (0, 0), (-1, -1), 0.4, colors.HexColor("#bdbdbd")),
        ("INNERGRID",    (0, 0), (-1, -1), 0.25, colors.HexColor("#dcdcdc")),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
    ]))
    return t


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------
def build_story():
    story = []

    # ---- Header ----
    story.append(P("Informe de Seguimiento #1", H1))
    story.append(P(
        "<b>Proyecto:</b> Segmentación Semántica con Imágenes Naturales (Project 3)<br/>"
        "<b>Asignatura:</b> Xarxes Neuronals i Aprenentatge Profund — Grau d'Enginyeria de Dades, UAB 2026<br/>"
        "<b>Grup:</b> 05<br/>"
        f"<b>Fecha:</b> {date.today().strftime('%d/%m/%Y')}",
        BODY,
    ))
    story.append(Spacer(1, 6))

    # ---- 1. Objetivo ----
    story.append(P("1. Objetivo del proyecto", H2))
    story.append(P(
        "Implementar un sistema de <b>segmentación semántica píxel a píxel</b> sobre imágenes "
        "naturales. Dada una imagen RGB de tamaño <i>(H&times;W&times;3)</i>, la red debe producir un mapa "
        "<i>(H&times;W)</i> donde cada píxel está asignado a una clase semántica. El objetivo final es entrenar "
        "y evaluar sobre <b>COCO 256&times;256</b>, partiendo de un baseline funcional sobre <b>PASCAL VOC 2012</b> "
        "(21 clases) por su disponibilidad inmediata vía <i>torchvision</i> y su tamaño manejable.",
        BODY,
    ))

    # ---- 2. Estado de partida ----
    story.append(P("2. Estado de partida y diagnóstico", H2))
    story.append(P(
        "El equipo arrancó con seis fragmentos de código (<i>bloque1</i>–<i>bloque6</i> y <i>encoder.py</i>) "
        "extraídos de la referencia técnica del curso. Tras inspeccionarlos se identificaron los siguientes "
        "problemas que impedían la ejecución:",
        BODY,
    ))
    story.extend(bullets([
        "Todos los archivos estaban envueltos en backticks markdown (``` ``` ```), por lo que no eran Python ejecutable.",
        "Faltaban imports en los seis archivos.",
        "<i>bloque3.py</i>: la <b>Dice loss</b> usaba una variable <code>C</code> indefinida (debía ser <code>preds.shape[1]</code>).",
        "<i>bloque4.py</i>: la función <code>validate()</code> no calculaba la loss, no reseteaba métricas y no movía las máscaras al device.",
        "<i>bloque5.py</i>: el método <code>update()</code> tenía un placeholder <code>...</code> sin implementar y <code>compute()</code> referenciaba variables inexistentes.",
        "<i>encoder.py</i>: la clase <code>UNet</code> no tenía <code>__init__</code> ni decoder real — no se podía instanciar.",
        "<i>bloque6.py</i>: la configuración usaba VOC2012/21 clases pero el enunciado pide COCO/256&times;256.",
    ]))

    # ---- 3. Modificaciones realizadas ----
    story.append(P("3. Modificaciones realizadas", H2))

    story.append(P("3.1. Reparación y reorganización de los archivos base", H3))
    story.append(P(
        "Se hicieron ejecutables los seis bloques y se renombraron siguiendo la estructura sugerida en "
        "<code>CLAUDE.md</code>:",
        BODY,
    ))
    rename_rows = [
        ["Archivo original", "Archivo final"],
        ["bloque1.py (Dataset)",       "dataset.py"],
        ["bloque3.py (Losses)",        "losses.py"],
        ["bloque4.py (train+validate)", "engine.py"],
        ["bloque5.py (Metrics)",       "metrics.py"],
        ["bloque6.py (Config)",        "config.py"],
        ["encoder.py (UNet)",          "models/unet.py"],
    ]
    story.append(two_col_table(rename_rows))
    story.append(Spacer(1, 6))

    story.append(P("3.2. Cambios técnicos clave", H3))
    story.extend(bullets([
        "<b>U-Net completa</b>: añadido <code>DecoderBlock</code> (ConvTranspose2d + concat skip + 2&times; Conv-BN-ReLU) y "
        "<code>UNet.__init__()</code> con head <code>Conv 1&times;1</code>. El forward aplica <code>F.interpolate</code> al "
        "final para garantizar que la salida tiene exactamente el mismo tamaño que la entrada (ResNet hace stride /32, "
        "no múltiplo limpio para 256).",
        "<b>Dice Loss</b>: ahora deduce <code>num_classes</code> dinámicamente de <code>preds.shape[1]</code>; añadido "
        "<code>ignore_index=255</code> en la Cross-Entropy combinada (convención VOC).",
        "<b>Métricas</b>: <code>SegmentationMetrics</code> implementada con <i>confusion matrix</i> acumulada vía "
        "<code>torch.bincount</code>; filtra el <code>ignore_index</code> y devuelve mIoU + IoU por clase. "
        "Se calcula sobre el epoch completo, no promediando por batch.",
        "<b>Engine</b>: <code>train_one_epoch()</code> acumula la loss; <code>validate()</code> ahora resetea métricas, "
        "calcula loss y métrica al mismo tiempo y mueve <i>todas</i> las tensores al device.",
        "<b>Config</b>: separados <code>LR_ENCODER=1e-5</code> y <code>LR_DECODER=1e-4</code> para no destruir los pesos "
        "preentrenados de ImageNet; <code>IMG_SIZE</code> fijado a 256 según enunciado.",
    ]))

    story.append(P("3.3. Componentes nuevos creados", H3))
    story.extend(bullets([
        "<b>main.py</b>: punto de entrada con CLI (<code>--overfit N</code>, <code>--epochs</code>, "
        "<code>--no-wandb</code>, <code>--data-root</code>). Orquesta dataset &rarr; modelo &rarr; loss &rarr; "
        "optimizer (parameter groups) &rarr; scheduler (CosineAnnealingLR) &rarr; engine. Loguea train/val loss, "
        "mIoU global y <b>IoU por clase</b> a Wandb. Guarda el checkpoint del mejor mIoU en "
        "<code>checkpoints/best.pt</code>.",
        "<b>transforms.py</b>: <code>PairedTransform</code> que aplica resize (bilinear en imagen, nearest en "
        "máscara), random horizontal flip sincronizado y normalización ImageNet. La máscara mantiene los índices "
        "de clase como <i>LongTensor</i>.",
        "<b>evaluate.py</b>: script independiente para cargar un checkpoint, calcular el mIoU final, listar el "
        "IoU por clase ordenado, y generar una figura cualitativa <i>imagen</i> | <i>GT</i> | <i>predicción</i> "
        "que se guarda en <code>docs/qualitative_results.png</code>.",
        "<b>classes.py</b>: nombres de clase y paleta de colores oficial de VOC2012 (compartidos por main.py y "
        "evaluate.py).",
        "<b>environment.yml</b>: actualizado a Python 3.10 con torch, torchvision, wandb, tqdm, matplotlib, "
        "pillow, numpy.",
        "<b>README.md</b>: documentación completa (objetivo, arquitectura, decisiones de diseño, instrucciones "
        "de reproducción, estado actual). Reemplaza la plantilla por defecto del template GitHub Classroom.",
    ]))

    story.append(P("3.4. Limpieza del template MNIST", H3))
    story.append(P(
        "El repositorio venía con un ejemplo CNN para MNIST. Se eliminaron los archivos no relevantes para "
        "segmentación: <code>train.py</code>, <code>test.py</code>, <code>test_trained_model.ipynb</code>, "
        "<code>models/models.py</code> y la carpeta <code>utils/</code> entera. Se conservan "
        "<code>LICENSE</code>, <code>.gitignore</code>, <code>.github/</code> y <code>test/</code> (este último "
        "contiene los checks de GitHub Classroom usados para la evaluación automática).",
        BODY,
    ))

    story.append(PageBreak())

    # ---- 4. Estructura final ----
    story.append(P("4. Estructura final del repositorio", H2))
    story.append(P(
        "<font face='Courier'>"
        "projecte-deep-learning-05/<br/>"
        "├── README.md<br/>"
        "├── LICENSE<br/>"
        "├── environment.yml<br/>"
        "├── main.py            &larr; punto de entrada — entrenamiento<br/>"
        "├── evaluate.py        &larr; evaluación cuantitativa + visualización cualitativa<br/>"
        "├── config.py          &larr; hiperparámetros<br/>"
        "├── classes.py         &larr; nombres de clase y paleta VOC<br/>"
        "├── dataset.py         &larr; SegmentationDataset (modo manual)<br/>"
        "├── transforms.py      &larr; transform sincronizado imagen-máscara<br/>"
        "├── losses.py          &larr; DiceLoss + SegmentationLoss<br/>"
        "├── metrics.py         &larr; SegmentationMetrics (mIoU)<br/>"
        "├── engine.py          &larr; train_one_epoch + validate (con tqdm)<br/>"
        "├── models/<br/>"
        "│   ├── __init__.py<br/>"
        "│   └── unet.py        &larr; Encoder ResNet50 + Decoder + UNet<br/>"
        "├── docs/<br/>"
        "│   ├── generate_report.py<br/>"
        "│   └── informe_seguimiento_1.pdf<br/>"
        "└── test/              &larr; checks Classroom (no tocar)"
        "</font>",
        CODE,
    ))

    # ---- 5. Arquitectura del baseline ----
    story.append(P("5. Arquitectura del baseline", H2))
    story.append(P(
        "Se ha implementado una <b>U-Net con encoder ResNet50 preentrenado en ImageNet</b>. La elección "
        "frente a VGG está justificada por dos razones: (i) ResNet no termina en capas fully-connected, "
        "preservando la información espacial necesaria para segmentación, y (ii) sus skip connections "
        "residuales internas mejoran el flujo de gradiente, lo cual es crítico al hacer fine-tuning.",
        BODY,
    ))
    story.append(P(
        "<font face='Courier'>"
        "Input (B, 3, 256, 256)<br/>"
        "    │<br/>"
        "    ▼<br/>"
        "[Encoder ResNet50 preentrenado]<br/>"
        "    │  layer0 (1/4)   ───────┐<br/>"
        "    │  layer1 (1/4)   ─────┐ │<br/>"
        "    │  layer2 (1/8)   ───┐ │ │<br/>"
        "    │  layer3 (1/16)  ─┐ │ │ │<br/>"
        "    ▼                  │ │ │ │<br/>"
        "[Bottleneck layer4 (1/32, 2048ch)]<br/>"
        "    │                  │ │ │ │<br/>"
        "    ▼                  │ │ │ │<br/>"
        "[Decoder: 4 × (Up + skip + 2× Conv-BN-ReLU)]<br/>"
        "    │  ◄───────────────┘ │ │ │<br/>"
        "    │  ◄─────────────────┘ │ │<br/>"
        "    │  ◄───────────────────┘ │<br/>"
        "    │  ◄─────────────────────┘<br/>"
        "    ▼<br/>"
        "[Conv 1×1 → num_classes] → bilinear → (B, num_classes, 256, 256)"
        "</font>",
        CODE,
    ))

    # ---- 6. Decisiones de diseño ----
    story.append(P("6. Decisiones de diseño", H2))
    decision_rows = [
        ["Decisión", "Razón"],
        ["ResNet50 como encoder",
         "Sin FC final; skip connections residuales internas; mejor flujo de gradiente que VGG."],
        ["Pesos ImageNet",
         "Transfer learning aprovecha features de bajo y medio nivel ya útiles."],
        ["LR_ENCODER (1e-5) ≪ LR_DECODER (1e-4)",
         "Encoder preentrenado: bajar LR evita destruir pesos. Decoder se entrena desde cero."],
        ["ConvTranspose2d (no Upsample)",
         "Upsampling con parámetros aprendibles → reconstrucción más precisa."],
        ["Loss combinada CE + Dice (0.5 / 0.5)",
         "CE penaliza correctamente píxel a píxel; Dice compensa el desbalanceo de clases."],
        ["IMG_SIZE = 256",
         "Lo pide el enunciado y reduce el coste de entrenamiento."],
        ["ignore_index = 255",
         "Convención VOC para los píxeles de borde no etiquetados."],
        ["VOC2012 antes que COCO",
         "Tamaño manejable, integración nativa con torchvision; baseline rápido para sanity check."],
    ]
    story.append(two_col_table(decision_rows, col_widths=(5.5 * cm, 11.5 * cm)))

    story.append(PageBreak())

    # ---- 7. Reproducibilidad ----
    story.append(P("7. Reproducibilidad", H2))
    story.append(P(
        "Pasos para reproducir el estado actual del proyecto:",
        BODY,
    ))
    story.append(P(
        "<font face='Courier'>"
        "git clone https://github.com/ED-2526/projecte-deep-learning-05.git<br/>"
        "cd projecte-deep-learning-05<br/>"
        "conda env create --file environment.yml<br/>"
        "conda activate xnap-segmentation<br/>"
        "wandb login<br/>"
        "<br/>"
        "# Sanity check: overfit con 5 imágenes (~ 30 epochs)<br/>"
        "python main.py --overfit 5 --epochs 30 --no-wandb<br/>"
        "<br/>"
        "# Entrenamiento completo VOC2012 (descarga ~2 GB la primera vez)<br/>"
        "python main.py --data-root ./data --epochs 50<br/>"
        "<br/>"
        "# Evaluación cuantitativa + figura cualitativa<br/>"
        "python evaluate.py --ckpt checkpoints/best.pt --num-samples 8"
        "</font>",
        CODE,
    ))

    # ---- 8. Checklist primera revisión ----
    story.append(P("8. Checklist de la primera revisión", H2))
    checklist_rows = [
        ["Tarea", "Estado"],
        ["Repositorio GitHub creado vía Classroom",            "OK"],
        ["Datasets / Dataloaders",                             "OK"],
        ["Pipeline ejecutable end-to-end (main.py)",           "OK"],
        ["Loss combinada (CE + Dice)",                         "OK"],
        ["Métrica mIoU implementada correctamente",            "OK"],
        ["U-Net con ResNet50 preentrenado",                    "OK"],
        ["Configuración con LRs separados encoder/decoder",    "OK"],
        ["Wandb integrado",                                    "OK"],
        ["Augmentaciones sincronizadas imagen-máscara",        "OK"],
        ["IoU por clase logueado a Wandb",                     "OK"],
        ["Script de evaluación + visualización cualitativa",   "OK"],
        ["Documentación (README, este informe)",               "OK"],
        ["environment.yml actualizado",                        "OK"],
        ["Sanity check de overfit ejecutado",                  "Pendiente (requiere GPU)"],
        ["Entrenamiento completo en VOC2012",                  "Pendiente"],
        ["Migración a COCO",                                   "Pendiente (2ª iteración)"],
    ]
    story.append(two_col_table(checklist_rows, col_widths=(11.0 * cm, 6.0 * cm)))

    # ---- 9. Preguntas para la siguiente sesión ----
    story.append(P("9. Preguntas para la siguiente sesión", H2))
    story.append(P(
        "Siguiendo la directriz del enunciado (\"On every question you should provide the evidences to "
        "answer it\"), planteamos las siguientes preguntas a responder con evidencias en la sesión #2:",
        BODY,
    ))
    story.extend(bullets([
        "¿El pipeline aprende correctamente? Evidencia: curva de loss en el sanity check de overfit con 5 imágenes; "
        "se espera que llegue a ~0.",
        "¿Cuál es el mIoU del baseline en VOC2012 val? Evidencia: tabla con mIoU global e IoU por clase tras 50 epochs.",
        "¿Cuáles son las clases con peor IoU? Evidencia: tabla ordenada por IoU para identificar dónde fallar el modelo.",
        "¿Hay overfitting? Evidencia: curvas <i>train_loss</i> vs. <i>val_loss</i> a lo largo del entrenamiento.",
        "¿Cuál es el impacto de la Dice loss frente a sólo Cross-Entropy? Evidencia: ablation con "
        "<code>DICE_WEIGHT=0</code> manteniendo el resto igual.",
    ]))

    # ---- 10. Próximos pasos ----
    story.append(P("10. Próximos pasos hacia la sesión #2", H2))
    story.extend(bullets([
        "Ejecutar el sanity check de overfit y capturar la curva de loss.",
        "Lanzar el entrenamiento completo en VOC2012 (50 epochs).",
        "Visualizar predicciones cualitativas (imagen / GT / predicción) en una figura comparativa.",
        "Preparar el dataset COCO 256&times;256 y adaptar <code>main.py</code> para soportar ambos.",
        "Empezar el ablation: probar variantes del modelo (encoder ResNet18 vs ResNet50, con vs sin pretraining).",
    ]))

    return story


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title="Informe Seguimiento #1 — Segmentación Semántica",
        author="Grup 05 — XNAP",
    )
    doc.build(build_story())
    print(f"Generated: {OUT}")


if __name__ == "__main__":
    main()
