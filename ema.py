"""
EMA (Exponential Moving Average) de los pesos del modelo.

Mantiene una copia "shadow" de los parámetros entrenables que se actualiza
cada step:
    shadow = decay * shadow + (1 - decay) * current

Para validar se swappean los pesos del modelo por los del shadow (más
suaves) y luego se restauran. Suele dar +1-2 pp de mIoU en val sin coste
en el train (excepto la copia adicional en VRAM).

Uso:
    ema = EMA(model, decay=0.9999)
    # en cada step de entrenamiento:
    optimizer.step()
    ema.update(model)
    # antes de validar:
    ema.apply_shadow(model)
    val_loss, val_metrics = validar(...)
    ema.restore(model)
"""
import torch


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        # solo parámetros entrenables; los buffers (BN running stats) NO se promedian
        self.shadow = {
            n: p.detach().clone()
            for n, p in model.named_parameters() if p.requires_grad
        }
        self._backup = None

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for n, p in model.named_parameters():
            if not p.requires_grad or n not in self.shadow:
                continue
            self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_shadow(self, model: torch.nn.Module) -> None:
        """Guarda los pesos actuales y los reemplaza con la versión EMA."""
        self._backup = {
            n: p.detach().clone()
            for n, p in model.named_parameters() if n in self.shadow
        }
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> None:
        """Restaura los pesos previos al apply_shadow."""
        if self._backup is None:
            return
        for n, p in model.named_parameters():
            if n in self._backup:
                p.copy_(self._backup[n])
        self._backup = None

    def state_dict(self) -> dict:
        return {n: t.detach().clone() for n, t in self.shadow.items()}

    def load_state_dict(self, sd: dict) -> None:
        for n, t in sd.items():
            if n in self.shadow:
                self.shadow[n].copy_(t)
