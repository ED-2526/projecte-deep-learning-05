"""
Plots y exports para el informe que W&B NO hace bien.

W&B ya cubre los training curves, bar charts de comparativas, scatter de
sweeps, etc. Configura paneles en el dashboard para esos. Este script solo
hace las dos cosas que W&B no resuelve:

  1. Confusion matrix heatmap (a partir del checkpoint, no de W&B).
  2. Export CSV de los runs registrados en runs.py (best_mIoU + métricas
     que pongas), listo para pegar en una tabla del informe.

Uso
---
    # 1) Confusion matrix de un checkpoint
    python report_plots.py cm --ckpt checkpoints/best.pt
    python report_plots.py cm --ckpt checkpoints/best.pt --normalize col --out docs/cm_precision.png

    # 2) Export CSV de un bloque de runs.py
    python report_plots.py csv --block loss_comparison
    python report_plots.py csv --block loss_comparison --out docs/loss_comparison.csv

    # 3) Export CSV de TODOS los bloques rellenos
    python report_plots.py csv --all
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch


# ═══════════════════════════════════════════════════════════════════════════
# 1. CONFUSION MATRIX HEATMAP
# ═══════════════════════════════════════════════════════════════════════════
def plot_confusion_matrix(ckpt_path: str, out_path: str,
                          normalize: str = "row",
                          annotate_threshold: int = 25,
                          cmap: str = "Blues",
                          mask_diagonal: bool = False) -> None:
    """
    Heatmap de la confusion matrix guardada en `ckpt['confusion_matrix']`.

    normalize: 'row' (recall por clase) | 'col' (precision por clase) | 'all' | 'none'
    annotate_threshold: si num_classes <= este número, escribe el % en cada celda.
    mask_diagonal: si True, oculta la diagonal (útil para ver SOLO confusiones).
    """
    import matplotlib.pyplot as plt

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "confusion_matrix" not in ckpt:
        raise ValueError(
            f"El checkpoint {ckpt_path} no contiene 'confusion_matrix'. "
            "Es de un entrenamiento previo a la modificación de main.py — "
            "necesitas reentrenar (el siguiente checkpoint la incluirá)."
        )
    cm     = ckpt["confusion_matrix"].float()
    names  = ckpt.get("class_names", [str(i) for i in range(cm.shape[0])])
    miou   = ckpt.get("mIoU", None)
    epoch  = ckpt.get("epoch", "?")
    n      = cm.shape[0]

    # Normalización
    if normalize == "row":
        cm_norm = cm / cm.sum(dim=1, keepdim=True).clamp(min=1)
        title_fmt = "Confusion matrix (filas normalizadas — recall por clase real)"
    elif normalize == "col":
        cm_norm = cm / cm.sum(dim=0, keepdim=True).clamp(min=1)
        title_fmt = "Confusion matrix (columnas normalizadas — precision por clase predicha)"
    elif normalize == "all":
        cm_norm = cm / cm.sum().clamp(min=1)
        title_fmt = "Confusion matrix (normalizada por total)"
    elif normalize == "none":
        cm_norm = cm
        title_fmt = "Confusion matrix (counts crudos)"
    else:
        raise ValueError(f"normalize debe ser row|col|all|none, no {normalize!r}")

    cm_np = cm_norm.numpy()
    if mask_diagonal:
        np.fill_diagonal(cm_np, np.nan)

    # Tamaño de figura proporcional al número de clases
    fig_side = max(8, n * 0.18)
    fig, ax = plt.subplots(figsize=(fig_side, fig_side))
    im = ax.imshow(cm_np, cmap=cmap, vmin=0, vmax=cm_np[~np.isnan(cm_np)].max() if mask_diagonal else None)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=90, fontsize=max(6, 10 - n // 10))
    ax.set_yticklabels(names, fontsize=max(6, 10 - n // 10))
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Ground truth")

    title = title_fmt
    if miou is not None:
        title += f"\n(mIoU = {miou:.4f}, epoch {epoch})"
    ax.set_title(title)

    if n <= annotate_threshold and normalize != "none":
        for i in range(n):
            for j in range(n):
                v = cm_np[i, j]
                if np.isnan(v):
                    continue
                color = "white" if v > 0.5 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=color, fontsize=7)

    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[report] confusion matrix guardada en {out}")


# ═══════════════════════════════════════════════════════════════════════════
# 2. EXPORT CSV DESDE W&B
# ═══════════════════════════════════════════════════════════════════════════
# Métricas que extraemos de cada run (clave en run.summary). Si una no existe,
# se rellena con vacío. Añade/quita aquí lo que quieras en la tabla del informe.
SUMMARY_KEYS = (
    "best_mIoU",
    "val_mIoU",
    "val_loss",
    "train_loss",
    "val_pixel_acc",
    "val_mF1",
    "val_boundary_mIoU",
)

# Hiperparámetros del config que también queremos en la tabla.
CONFIG_KEYS = (
    "BACKBONE",
    "EPOCHS",
    "OPTIMIZER",
    "LR_DECODER",
    "CE_WEIGHT", "DICE_WEIGHT", "FOCAL_WEIGHT",
    "LOVASZ_WEIGHT", "OHEM_CE_WEIGHT", "WEIGHTED_CE_WEIGHT",
    "FOCAL_GAMMA", "OHEM_TOP_K",
)


def _fetch_run(api, entity, project, run_id):
    """Devuelve el objeto run o None si no existe / falla."""
    try:
        return api.run(f"{entity}/{project}/{run_id}")
    except Exception as e:
        print(f"  [WARN] no pude cargar {run_id}: {e}")
        return None


def export_block_csv(block: str, out_path: str) -> None:
    """Exporta un bloque entero de runs.py a un CSV con métricas + config."""
    import wandb
    from runs import EXPERIMENTS, WANDB_ENTITY, WANDB_PROJECT

    if block not in EXPERIMENTS:
        raise KeyError(f"Bloque {block!r} no existe en runs.py. "
                       f"Disponibles: {list(EXPERIMENTS)}")

    api = wandb.Api()
    rows = []
    print(f"[report] exportando bloque '{block}'...")
    for name, info in EXPERIMENTS[block].items():
        rid = info["run_id"]
        if not rid:
            print(f"  [SKIP] {name}: run_id vacío")
            continue
        run = _fetch_run(api, WANDB_ENTITY, WANDB_PROJECT, rid)
        if run is None:
            continue
        row = {
            "block":       block,
            "name":        name,
            "label":       info.get("label", name),
            "run_id":      rid,
            "wandb_url":   f"https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}/runs/{rid}",
            "state":       run.state,
        }
        for k in SUMMARY_KEYS:
            v = run.summary.get(k, "")
            row[k] = float(v) if isinstance(v, (int, float)) else v
        for k in CONFIG_KEYS:
            row[f"cfg.{k}"] = run.config.get(k, "")
        rows.append(row)
        print(f"  ✓ {name:<22s} best_mIoU={row.get('best_mIoU', 'N/A')}")

    if not rows:
        print(f"[report] sin runs rellenos en bloque {block!r}; no se genera CSV")
        return

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[report] CSV guardado en {out}  ({len(rows)} runs)")


def export_all_csv(out_dir: str) -> None:
    """Exporta TODOS los bloques rellenos de runs.py, uno por CSV."""
    from runs import EXPERIMENTS
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for block in EXPERIMENTS:
        export_block_csv(block, out_dir / f"{block}.csv")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description="Plots y exports para el informe.")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Confusion matrix
    p_cm = sub.add_parser("cm", help="Heatmap de la confusion matrix de un checkpoint")
    p_cm.add_argument("--ckpt", type=str, default="checkpoints/best.pt")
    p_cm.add_argument("--out",  type=str, default="docs/confusion_matrix.png")
    p_cm.add_argument("--normalize", type=str, default="row",
                      choices=("row", "col", "all", "none"))
    p_cm.add_argument("--mask-diagonal", action="store_true",
                      help="Pone NaN en la diagonal para ver solo las confusiones")
    p_cm.add_argument("--cmap", type=str, default="Blues",
                      help="Colormap de matplotlib (Blues, Reds, viridis, ...)")

    # CSV export
    p_csv = sub.add_parser("csv", help="Export CSV de un bloque (o todos) de runs.py")
    p_csv.add_argument("--block", type=str, default=None,
                       help="Nombre de bloque en runs.py (ej. loss_comparison)")
    p_csv.add_argument("--all",   action="store_true",
                       help="Exporta TODOS los bloques (un CSV por bloque)")
    p_csv.add_argument("--out",   type=str, default=None,
                       help="Ruta de salida (CSV con --block, dir con --all)")

    args = p.parse_args()

    if args.cmd == "cm":
        plot_confusion_matrix(args.ckpt, args.out, normalize=args.normalize,
                              cmap=args.cmap, mask_diagonal=args.mask_diagonal)
    elif args.cmd == "csv":
        if args.all:
            out_dir = args.out or "docs/csv"
            export_all_csv(out_dir)
        elif args.block:
            out = args.out or f"docs/{args.block}.csv"
            export_block_csv(args.block, out)
        else:
            p.error("usa --block <nombre> o --all")


if __name__ == "__main__":
    main()
