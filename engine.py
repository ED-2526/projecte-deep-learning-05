import torch
from tqdm.auto import tqdm


def train_one_epoch(model, loader, optimizer, criterion, device, epoch=None):
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
def validate(model, loader, criterion, metrics, device, epoch=None):
    model.eval()
    metrics.reset()
    total_loss = 0.0
    desc = f"val   ep{epoch:03d}" if epoch is not None else "val"
    pbar = tqdm(loader, desc=desc, leave=False)

    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        preds = model(images)
        total_loss += criterion(preds, masks).item()
        metrics.update(preds, masks)
        pbar.set_postfix(loss=f"{total_loss/(pbar.n+1):.4f}")

    return total_loss / max(len(loader), 1), metrics.compute()
