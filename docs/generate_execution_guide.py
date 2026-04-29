"""Genera el PDF de la guía de ejecución (entorno virtual + comandos).

Uso:
    python docs/generate_execution_guide.py

Salida:
    docs/guia_ejecucion.pdf
"""
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, PageBreak
)


OUT = Path(__file__).parent / "guia_ejecucion.pdf"

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=18, spaceAfter=14,
                    textColor=colors.HexColor("#1a1a1a"))
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=13, spaceBefore=14, spaceAfter=8,
                    textColor=colors.HexColor("#1a1a1a"))
H3 = ParagraphStyle("H3", parent=ss["Heading3"], fontSize=11, spaceBefore=8, spaceAfter=4,
                    textColor=colors.HexColor("#333333"))
BODY = ParagraphStyle("Body", parent=ss["BodyText"], fontSize=10, leading=14,
                      alignment=TA_JUSTIFY, spaceAfter=6)
BULLET = ParagraphStyle("Bullet", parent=BODY, leftIndent=14, bulletIndent=4,
                        spaceAfter=2, alignment=TA_LEFT)
CODE = ParagraphStyle("Code", parent=ss["Code"], fontSize=9, leading=11,
                      backColor=colors.HexColor("#f5f5f5"),
                      borderColor=colors.HexColor("#dcdcdc"), borderWidth=0.5,
                      borderPadding=6, spaceAfter=8)


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


def build_story():
    story = []

    # ---- Header ----
    story.append(P("Guía de Implementación y Ejecución", H1))
    story.append(P(
        "<b>Proyecto:</b> Segmentación Semántica con Imágenes Naturales (Project 3)<br/>"
        "<b>Asignatura:</b> Xarxes Neuronals i Aprenentatge Profund — Grau d'Enginyeria de Dades, UAB 2026<br/>"
        "<b>Grup:</b> 05<br/>"
        f"<b>Fecha:</b> {date.today().strftime('%d/%m/%Y')}<br/>"
        "<b>Repositorio:</b> https://github.com/ED-2526/projecte-deep-learning-05",
        BODY,
    ))
    story.append(Spacer(1, 6))

    story.append(P(
        "Este documento complementa el <i>Informe de Seguimiento #1</i>. Recoge: "
        "(1) las mejoras añadidas tras el primer informe, (2) el procedimiento exacto para crear el entorno "
        "virtual con GPU y (3) los comandos de ejecución y las salidas esperadas.",
        BODY,
    ))

    # =========================================================
    # 1. Mejoras añadidas
    # =========================================================
    story.append(P("1. Mejoras añadidas en esta iteración", H2))
    story.append(P(
        "Sobre el baseline ya entregable, se han añadido los siguientes elementos para que el entorno con GPU "
        "pueda ejecutar el flujo completo (entrenamiento + evaluación + visualización) sin tocar código:",
        BODY,
    ))

    story.append(P("1.1. Instrumentación del entrenamiento", H3))
    story.extend(bullets([
        "<b>tqdm en engine.py</b>: las funciones <code>train_one_epoch()</code> y <code>validate()</code> "
        "muestran ahora barras de progreso con la loss del batch actual. Imprescindible para monitorizar runs "
        "largos en remoto.",
        "<b>Logging de IoU por clase a Wandb</b>: <code>main.py</code> construye un dict "
        "<code>val_iou/&lt;nombre_clase&gt;</code> con las 21 clases de VOC y lo loguea en cada epoch. "
        "Permite ver en el dashboard qué clases progresan y cuáles se atascan.",
        "<b>Checkpoint con metadata</b>: <code>best.pt</code> guarda ahora un dict con "
        "<code>{epoch, model_state_dict, mIoU, config}</code>, no sólo los pesos. Esto evita ambigüedad al "
        "cargar el modelo y permite trazar resultados.",
    ]))

    story.append(P("1.2. Componentes nuevos", H3))
    story.extend(bullets([
        "<b>classes.py</b>: nombres de clase de VOC2012 (<code>VOC_CLASSES</code>, 21 nombres) y la paleta "
        "oficial de colores (<code>VOC_COLORMAP</code>) usada para colorear las máscaras al visualizar.",
        "<b>evaluate.py</b>: script independiente para validación final. Carga un checkpoint, calcula el mIoU y "
        "el IoU por clase ordenados de peor a mejor, y genera <code>docs/qualitative_results.png</code> con N "
        "ejemplos en formato <i>imagen | ground truth | predicción</i>. Diseñado para responder a la pregunta "
        "\"¿qué clases falla más el modelo?\" en la siguiente sesión.",
        "<b>models/__init__.py</b>: ahora re-exporta <code>UNet</code>, <code>Encoder</code> y "
        "<code>DecoderBlock</code> para imports más limpios.",
    ]))

    story.append(P("1.3. Robustez del flujo de overfit", H3))
    story.append(P(
        "Se ha simplificado el modo <code>--overfit N</code>: ahora <code>val_ds</code> apunta directamente al "
        "<code>train_ds</code> recortado, en lugar de crear un Subset anidado. La intención es que las mismas N "
        "imágenes se usen para entrenar y validar; si la loss baja a casi cero y el mIoU se acerca a 1, el "
        "pipeline funciona.",
        BODY,
    ))

    # =========================================================
    # 2. Estructura final del repositorio
    # =========================================================
    story.append(P("2. Estructura final del repositorio", H2))
    story.append(P(
        "<font face='Courier'>"
        "projecte-deep-learning-05/<br/>"
        "├── README.md            documentación principal<br/>"
        "├── LICENSE<br/>"
        "├── environment.yml      entorno conda<br/>"
        "├── main.py              entrenamiento (CLI)<br/>"
        "├── evaluate.py          evaluación cuantitativa + figura cualitativa<br/>"
        "├── config.py            hiperparámetros<br/>"
        "├── classes.py           nombres y paleta VOC<br/>"
        "├── dataset.py           SegmentationDataset (modo manual)<br/>"
        "├── transforms.py        PairedTransform (sincronizado img-mask)<br/>"
        "├── losses.py            DiceLoss + SegmentationLoss<br/>"
        "├── metrics.py           SegmentationMetrics (mIoU + IoU/clase)<br/>"
        "├── engine.py            train_one_epoch + validate (con tqdm)<br/>"
        "├── models/<br/>"
        "│   ├── __init__.py<br/>"
        "│   └── unet.py          Encoder ResNet50 + Decoder + UNet<br/>"
        "├── docs/<br/>"
        "│   ├── generate_report.py<br/>"
        "│   ├── informe_seguimiento_1.pdf<br/>"
        "│   ├── generate_execution_guide.py<br/>"
        "│   └── guia_ejecucion.pdf      &larr; este documento<br/>"
        "└── test/                 checks Classroom (no tocar)"
        "</font>",
        CODE,
    ))

    story.append(PageBreak())

    # =========================================================
    # 3. Configuración del entorno virtual
    # =========================================================
    story.append(P("3. Configuración del entorno virtual con GPU", H2))

    story.append(P("3.1. Requisitos previos", H3))
    story.extend(bullets([
        "Driver NVIDIA actualizado y compatible con CUDA 11.8 o superior.",
        "Conda (Miniconda o Anaconda) instalado.",
        "Git instalado (en la máquina local ya está; en el remoto, instalar si falta).",
        "~5 GB de espacio libre (entorno + dataset VOC2012 + checkpoints).",
    ]))

    story.append(P("3.2. Clonar el repositorio", H3))
    story.append(P(
        "<font face='Courier'>"
        "git clone https://github.com/ED-2526/projecte-deep-learning-05.git<br/>"
        "cd projecte-deep-learning-05"
        "</font>",
        CODE,
    ))

    story.append(P("3.3. Crear el entorno conda", H3))
    story.append(P(
        "<font face='Courier'>"
        "conda env create --file environment.yml<br/>"
        "conda activate xnap-segmentation"
        "</font>",
        CODE,
    ))
    story.append(P(
        "El archivo <code>environment.yml</code> declara Python 3.10 con torch, torchvision, wandb, tqdm, "
        "matplotlib, pillow y numpy. Si la versión por defecto de PyTorch en pip no incluye soporte CUDA en tu "
        "sistema, fuerza la versión adecuada antes de continuar:",
        BODY,
    ))
    story.append(P(
        "<font face='Courier'>"
        "# Ejemplo para CUDA 12.1<br/>"
        "pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu121"
        "</font>",
        CODE,
    ))

    story.append(P("3.4. Verificar la instalación", H3))
    story.append(P(
        "<font face='Courier'>"
        "python -c \"import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')\""
        "</font>",
        CODE,
    ))
    story.append(P(
        "Salida esperada: <code>cuda: True NVIDIA &lt;modelo&gt;</code>. Si imprime <code>False</code> revisa el "
        "driver / la versión de torch.",
        BODY,
    ))

    story.append(P("3.5. Login en Wandb (opcional pero recomendable)", H3))
    story.append(P(
        "<font face='Courier'>"
        "wandb login"
        "</font>",
        CODE,
    ))
    story.append(P(
        "Si no quieres usar Wandb, todos los comandos aceptan <code>--no-wandb</code>. Para ejecutar pero "
        "subir más tarde, usa <code>--wandb-offline</code> y luego <code>wandb sync ./wandb/&lt;run&gt;</code>.",
        BODY,
    ))

    # =========================================================
    # 4. Ejecución
    # =========================================================
    story.append(P("4. Comandos de ejecución", H2))

    story.append(P("4.1. Sanity check — overfit con 5 imágenes", H3))
    story.append(P(
        "Demuestra que el pipeline aprende. Es la primera verificación recomendada por el enunciado del proyecto "
        "(\"Demonstrate that the model learns as expected\").",
        BODY,
    ))
    story.append(P(
        "<font face='Courier'>"
        "python main.py --overfit 5 --epochs 30 --no-wandb"
        "</font>",
        CODE,
    ))
    story.append(P(
        "<b>Esperable:</b> en CUDA &lt; 5 minutos. La <i>train_loss</i> debe bajar a casi 0 y el <i>val_mIoU</i> "
        "debe acercarse a 1 (las mismas imágenes en train y val).",
        BODY,
    ))

    story.append(P("4.2. Entrenamiento completo en VOC2012", H3))
    story.append(P(
        "<font face='Courier'>"
        "python main.py --data-root ./data --epochs 50"
        "</font>",
        CODE,
    ))
    story.append(P(
        "La primera vez descarga VOC2012 (~2 GB) en <code>./data</code>. Tiempo estimado en una GPU consumer "
        "(RTX 3060/4060): ~2–4 horas para 50 epochs con <code>BATCH_SIZE=8</code> e <code>IMG_SIZE=256</code>. "
        "El mejor modelo se guarda en <code>checkpoints/best.pt</code>.",
        BODY,
    ))

    story.append(P("4.3. Evaluación final + figura cualitativa", H3))
    story.append(P(
        "<font face='Courier'>"
        "python evaluate.py --ckpt checkpoints/best.pt --num-samples 8"
        "</font>",
        CODE,
    ))
    story.append(P(
        "Imprime el mIoU final y el IoU por clase ordenado de peor a mejor. Genera "
        "<code>docs/qualitative_results.png</code> con 8 ejemplos en formato <i>imagen | GT | predicción</i>.",
        BODY,
    ))

    # =========================================================
    # 5. Argumentos disponibles
    # =========================================================
    story.append(P("5. Argumentos de las CLIs", H2))

    story.append(P("main.py", H3))
    args_main = [
        ["Flag",                "Descripción"],
        ["--data-root",         "Carpeta donde descargar/leer VOC2012 (default: ./data)"],
        ["--epochs",            "Sobrescribe Config.EPOCHS"],
        ["--overfit N",         "Si > 0: entrena/valida sobre las N primeras imágenes (sanity check)"],
        ["--no-wandb",          "Desactiva el logging a Wandb"],
        ["--wandb-offline",     "Wandb en modo offline (sincronizar luego con wandb sync)"],
    ]
    story.append(two_col_table(args_main, col_widths=(4.5 * cm, 12.5 * cm)))

    story.append(Spacer(1, 6))
    story.append(P("evaluate.py", H3))
    args_eval = [
        ["Flag",                "Descripción"],
        ["--ckpt",              "Ruta del checkpoint a cargar (default: checkpoints/best.pt)"],
        ["--data-root",         "Carpeta de VOC2012 (default: ./data)"],
        ["--num-samples",       "Número de ejemplos en la figura cualitativa (default: 8)"],
        ["--no-figure",         "No genera la figura, sólo imprime las métricas"],
    ]
    story.append(two_col_table(args_eval, col_widths=(4.5 * cm, 12.5 * cm)))

    story.append(PageBreak())

    # =========================================================
    # 6. Salidas esperadas
    # =========================================================
    story.append(P("6. Salidas y dónde se guardan", H2))
    outputs = [
        ["Ruta",                                  "Contenido"],
        ["data/VOCdevkit/VOC2012/",               "Dataset VOC2012 descargado"],
        ["checkpoints/best.pt",                   "Mejor checkpoint (epoch, state_dict, mIoU, config)"],
        ["wandb/run-<id>/",                       "Logs locales de Wandb (sólo si se usa)"],
        ["docs/qualitative_results.png",          "Figura imagen | GT | predicción tras evaluate.py"],
        ["stdout",                                "Loss/mIoU por epoch + IoU por clase ordenado al evaluar"],
    ]
    story.append(two_col_table(outputs, col_widths=(7.0 * cm, 10.0 * cm)))

    # =========================================================
    # 7. Troubleshooting
    # =========================================================
    story.append(P("7. Problemas comunes y soluciones", H2))
    trouble = [
        ["Síntoma", "Causa probable / solución"],
        ["torch.cuda.is_available() = False",
         "Driver NVIDIA o versión de torch sin CUDA. Reinstalar torch con el wheel de la CUDA correspondiente."],
        ["CUDA out of memory",
         "Reducir BATCH_SIZE en config.py (de 8 a 4). Si persiste, reducir IMG_SIZE a 224."],
        ["Descarga de VOC2012 muy lenta o falla",
         "Repetir el comando — torchvision reanuda. Alternativa: descargar manualmente desde el mirror oficial."],
        ["Wandb pide login en cada ejecución",
         "Ejecutar <code>wandb login</code> una vez. Para CI/headless: variable WANDB_API_KEY."],
        ["Loss = NaN tras pocos batches",
         "LR demasiado alto o gradientes explotando. Revisar Config.LR_DECODER (1e-4 es seguro)."],
        ["mIoU congelado en ~0.05",
         "Posible desbalanceo extremo. Comprobar que los IDs de máscara están bien (max debe ser 20 o 255)."],
    ]
    story.append(two_col_table(trouble, col_widths=(5.0 * cm, 12.0 * cm)))

    # =========================================================
    # 8. Checklist de la primera revisión
    # =========================================================
    story.append(P("8. Checklist post-ejecución para la primera revisión", H2))
    story.append(P(
        "Tras correr los comandos anteriores en el entorno con GPU, deberías tener evidencias para defender lo "
        "siguiente:",
        BODY,
    ))
    story.extend(bullets([
        "<b>El pipeline aprende</b>: curva de loss del run de overfit (debe llegar a casi 0).",
        "<b>El modelo entrena</b>: curvas de train_loss y val_loss en Wandb del run completo.",
        "<b>Métrica cuantitativa</b>: mIoU final reportado por <code>evaluate.py</code>.",
        "<b>Análisis por clase</b>: lista de IoU por clase ordenada — identifica las clases más difíciles.",
        "<b>Análisis cualitativo</b>: figura <code>qualitative_results.png</code> con 8 ejemplos.",
        "<b>Evidencia de overfit/underfit</b>: comparación de train_loss vs. val_loss.",
    ]))

    return story


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title="Guía de Ejecución — Segmentación Semántica",
        author="Grup 05 — XNAP",
    )
    doc.build(build_story())
    print(f"Generated: {OUT}")


if __name__ == "__main__":
    main()
