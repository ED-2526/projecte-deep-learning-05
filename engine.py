import torch
from tqdm.auto import tqdm
from torch.cuda.amp import autocast, GradScaler


def entrenar_una_epoca(model, loader, optimizer, criterion, device, scheduler=None, epoch=None, use_amp=True):
    """
    EXPLICACIÓ SIMPLE: Entrena el model durant un epoch amb Mixed Precision Training (AMP).
    Per a cada batch:
    1. Passa imatges pel model (float16 para velocidad)
    2. Calcula l'error (pèrdua) en float32 para estabilidad
    3. Actualitza els pesos del model
    4. Actualitza el scheduler (warmup + cosine annealing)
    5. Mostra la pèrdua actual
    Mixed Precision = 1.5-2x más rápido sin perder precisión.
    """
    model.train()
    total_loss = 0.0
    desc = f"train ep{epoch:03d}" if epoch is not None else "train"
    pbar = tqdm(loader, desc=desc, leave=False)
    
    scaler = GradScaler() if use_amp and device.type == 'cuda' else None

    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        
        if scaler is not None:
            # Mixed precision: forward pass en float16
            with autocast(dtype=torch.float16):
                preds = model(images)
                loss = criterion(preds, masks)
            
            # Backward con scaling
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Sin AMP (CPU o GPU sin soporte)
            preds = model(images)
            loss = criterion(preds, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Actualizar scheduler por batch (para warmup)
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validar(model, loader, criterion, metrics, device, epoch=None):
    """
    EXPLICACIÓ SIMPLE: Avalua el model en dades de validació (sense actualitzar pesos).
    Per a cada batch:
    1. Passa imatges pel model
    2. Calcula l'error
    3. Actualitza les mètriques de precisió
    Retorna la pèrdua promitjada i les mètriques calculades (mIoU, IoU per classe).
    """
    model.eval()
    metrics.reinicialitzar()
    total_loss = 0.0
    desc = f"val   ep{epoch:03d}" if epoch is not None else "val"
    pbar = tqdm(loader, desc=desc, leave=False)

    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        preds = model(images)
        total_loss += criterion(preds, masks).item()
        metrics.actualitzar(preds, masks)
        pbar.set_postfix(loss=f"{total_loss/(pbar.n+1):.4f}")

    return total_loss / max(len(loader), 1), metrics.calcular()
