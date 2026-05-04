import torch
from tqdm.auto import tqdm


def entrenar_una_epoca(model, loader, optimizer, criterion, device, epoch=None):
    """
    EXPLICACIÓ SIMPLE: Entrena el model durant un epoch (una passada per totes les dades).
    Per a cada batch:
    1. Passa imatges pel model
    2. Calcula l'error (pèrdua)
    3. Actualitza els pesos del model
    4. Mostra la pèrdua actual
    Retorna la pèrdua promitjada de l'epoch.
    """
    model.train()
    total_loss = 0.0
    desc = f"train ep{epoch:03d}" if epoch is not None else "train"
    pbar = tqdm(loader, desc=desc, leave=False)

    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        preds = model(images)
        loss  = criterion(preds, masks)
        loss.backward()
        optimizer.step()

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
