import torch
from tqdm.auto import tqdm


def entrenar_una_epoca(model, loader, optimizer, criterion, device,
                       scaler=None, use_amp=False, channels_last=False, epoch=None):
    """
    EXPLICACIÓ SIMPLE: Entrena el model durant un epoch (una passada per totes les dades).
    Per a cada batch:
    1. Passa imatges pel model (en mixed precision si use_amp=True)
    2. Calcula l'error (pèrdua) en fp32
    3. Actualitza els pesos del model (amb GradScaler si AMP)
    4. Mostra la pèrdua actual
    Retorna la pèrdua promitjada de l'epoch.
    """
    model.train()
    total_loss = 0.0
    desc = f"train ep{epoch:03d}" if epoch is not None else "train"
    pbar = tqdm(loader, desc=desc, leave=False)
    mem_fmt = torch.channels_last if channels_last else torch.contiguous_format

    for images, masks in pbar:
        images = images.to(device, non_blocking=True, memory_format=mem_fmt)
        masks  = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            preds = model(images)
        loss = criterion(preds.float(), masks)   # la loss siempre en fp32 (Dice usa smooth=1e-6)

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validar(model, loader, criterion, metrics, device,
            use_amp=False, channels_last=False, epoch=None):
    """
    EXPLICACIÓ SIMPLE: Avalua el model en dades de validació (sense actualitzar pesos).
    Per a cada batch:
    1. Passa imatges pel model (mixed precision si use_amp=True)
    2. Calcula l'error en fp32
    3. Actualitza les mètriques de precisió
    Retorna la pèrdua promitjada i les mètriques calculades (mIoU, IoU per classe).
    """
    model.eval()
    metrics.reinicialitzar()
    total_loss = 0.0
    desc = f"val   ep{epoch:03d}" if epoch is not None else "val"
    pbar = tqdm(loader, desc=desc, leave=False)
    mem_fmt = torch.channels_last if channels_last else torch.contiguous_format

    for images, masks in pbar:
        images = images.to(device, non_blocking=True, memory_format=mem_fmt)
        masks  = masks.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            preds = model(images)
        total_loss += criterion(preds.float(), masks).item()
        metrics.actualitzar(preds, masks)
        pbar.set_postfix(loss=f"{total_loss/(pbar.n+1):.4f}")

    return total_loss / max(len(loader), 1), metrics.calcular()
