"""DeepLabV3+ con encoder ResNet (dilated) para segmentación semántica.

Diferencias clave respecto a la U-Net del proyecto:
- El encoder usa **atrous (dilated) convolutions** en layer4 (y opcionalmente
  también en layer3) para mantener mayor resolución espacial al final del
  encoder. Output stride (OS) habitual:
      OS = 16 → solo layer4 dilatada (típico, mejor compromiso vel/calidad)
      OS = 8  → layer3 y layer4 dilatadas (más memoria, ~+1pp mIoU)
- Sobre los features de layer4 va el módulo **ASPP** (Atrous Spatial Pyramid
  Pooling): 1×1 conv + tres 3×3 convs con dilations distintas + global pooling,
  concatenadas y proyectadas → captura contexto multi-escala sin reducir resolución.
- El **decoder** sube ASPP al stride 4 (resolución de layer1), concatena con
  los features "low-level" de layer1 (proyectados a 48 canales para no dominar),
  aplica 2× conv 3×3 + head 1×1 y hace bilinear upsample a la resolución original.

Encoder reutilizado del proyecto (`models/unet.Encoder`): mismo backbone, mismas
capas, mismos canales — solo se modifican los strides de las últimas capas para
introducir las dilations (vía `torchvision`'s `replace_stride_with_dilation`).

Salida: logits `(B, num_classes, H, W)` — el mismo formato que U-Net,
intercambiables en el resto del pipeline.

Referencia: Chen et al., "Encoder-Decoder with Atrous Separable Convolution for
Semantic Image Segmentation", ECCV 2018.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .unet import _BACKBONE_BUILDERS, _BACKBONE_CHANNELS


# ╭───────────────────────────────────────────────────────────────────────╮
# │ ASPP                                                                  │
# ╰───────────────────────────────────────────────────────────────────────╯

class _ASPPConv(nn.Sequential):
    """Conv 3×3 con dilation > 1 (rama atrous del ASPP)."""
    def __init__(self, in_ch: int, out_ch: int, dilation: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class _ASPPPooling(nn.Module):
    """Rama "image pooling" del ASPP: global average pool + 1×1 conv + upsample
    al tamaño espacial original."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        x = self.proj(self.pool(x))
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling.

    5 ramas paralelas (1×1 conv, tres 3×3 con dilations distintas, image-pooling),
    concatenadas en canal y proyectadas con 1×1 conv + dropout.

    `atrous_rates` típicos:
      output_stride 16  → (6, 12, 18)
      output_stride 8   → (12, 24, 36)
    """
    def __init__(self, in_ch: int, out_ch: int = 256,
                 atrous_rates=(6, 12, 18), dropout: float = 0.1):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ),
            *[_ASPPConv(in_ch, out_ch, rate) for rate in atrous_rates],
            _ASPPPooling(in_ch, out_ch),
        ])
        self.project = nn.Sequential(
            nn.Conv2d(len(self.branches) * out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [b(x) for b in self.branches]
        return self.project(torch.cat(feats, dim=1))


# ╭───────────────────────────────────────────────────────────────────────╮
# │ Encoder dilatado                                                      │
# ╰───────────────────────────────────────────────────────────────────────╯

class DilatedEncoder(nn.Module):
    """
    Encoder ResNet con strides reemplazados por dilations en las últimas capas,
    para mantener resolución espacial al final del encoder.

      output_stride = 16  → layer4 dilatada (dilation=2)
      output_stride = 8   → layer3 (dilation=2) y layer4 (dilation=4)

    Esto se hace con `replace_stride_with_dilation` de torchvision, que está
    pensado precisamente para este uso (DeepLab).

    forward → (feat_high, feat_low) donde:
      feat_high : output del layer4 (OS = 16 u 8)
      feat_low  : output del layer1 (OS = 4, usado por el decoder de DeepLabV3+)
    """
    def __init__(self, backbone: str = "resnet50", pretrained: bool = True,
                 output_stride: int = 16):
        super().__init__()
        if backbone not in _BACKBONE_BUILDERS:
            raise ValueError(
                f"Backbone '{backbone}' no soportado. "
                f"Opciones: {list(_BACKBONE_BUILDERS.keys())}"
            )
        if output_stride not in (8, 16):
            raise ValueError(f"output_stride debe ser 8 o 16, recibido {output_stride}")

        # En torchvision, replace_stride_with_dilation = [layer2, layer3, layer4].
        # Para OS=16 dilatamos solo layer4; para OS=8, también layer3.
        rswd = [False, False, True] if output_stride == 16 else [False, True, True]
        builder, weights_enum = _BACKBONE_BUILDERS[backbone]
        weights = weights_enum if pretrained else None
        net = builder(weights=weights, replace_stride_with_dilation=rswd)

        self.layer0 = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        # Canales por nivel (los mismos que la U-Net; solo cambia la resolución espacial)
        self.channels = _BACKBONE_CHANNELS[backbone]
        self.output_stride = output_stride

    def forward(self, x: torch.Tensor):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)               # low-level: OS = 4
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)               # high-level: OS = output_stride
        return x4, x1


# ╭───────────────────────────────────────────────────────────────────────╮
# │ DeepLabV3+                                                             │
# ╰───────────────────────────────────────────────────────────────────────╯

class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ con encoder ResNet dilatado. Salida: logits (B, num_classes, H, W).

    Args
    ----
    num_classes        Nº de clases del problema.
    backbone           "resnet18" | "resnet34" | "resnet50" | "resnet101" | "resnet152"
    pretrained         Cargar pesos de ImageNet.
    output_stride      16 (default, recomendado) o 8 (más memoria, ~+1pp mIoU).
    aspp_out_channels  Canales de salida del ASPP (256 estándar).
    decoder_low_ch     Canales de proyección de las low-level features (48 estándar;
                       valores grandes hacen que dominen sobre las high-level).
    decoder_out_ch     Canales internos del decoder.
    decoder_dropout    Dropout2d al final del decoder (consistente con UNet).
    aspp_dropout       Dropout aplicado dentro del módulo ASPP (default 0.1).
    """
    def __init__(self, num_classes: int, backbone: str = "resnet50",
                 pretrained: bool = True, output_stride: int = 16,
                 aspp_out_channels: int = 256, decoder_low_ch: int = 48,
                 decoder_out_ch: int = 256, decoder_dropout: float = 0.0,
                 aspp_dropout: float = 0.1):
        super().__init__()
        self.encoder = DilatedEncoder(backbone=backbone, pretrained=pretrained,
                                      output_stride=output_stride)
        high_ch = self.encoder.channels[0]      # canales de layer4
        low_ch  = self.encoder.channels[3]      # canales de layer1

        atrous_rates = (6, 12, 18) if output_stride == 16 else (12, 24, 36)
        self.aspp = ASPP(in_ch=high_ch, out_ch=aspp_out_channels,
                         atrous_rates=atrous_rates, dropout=aspp_dropout)

        # 1×1 conv para reducir los canales de las low-level features (estándar)
        self.low_project = nn.Sequential(
            nn.Conv2d(low_ch, decoder_low_ch, 1, bias=False),
            nn.BatchNorm2d(decoder_low_ch),
            nn.ReLU(inplace=True),
        )

        # Decoder: concat (ASPP up + low) → 2× conv 3×3 → head
        decoder_layers = [
            nn.Conv2d(aspp_out_channels + decoder_low_ch, decoder_out_ch, 3,
                      padding=1, bias=False),
            nn.BatchNorm2d(decoder_out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_out_ch, decoder_out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_out_ch),
            nn.ReLU(inplace=True),
        ]
        if decoder_dropout and decoder_dropout > 0.0:
            decoder_layers.append(nn.Dropout2d(p=decoder_dropout))
        self.decoder = nn.Sequential(*decoder_layers)
        self.head = nn.Conv2d(decoder_out_ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_size = x.shape[-2:]
        x_high, x_low = self.encoder(x)                         # high: OS=16, low: OS=4

        x_aspp = self.aspp(x_high)
        # Sube ASPP a la resolución de las low-level features (OS=4)
        x_aspp = F.interpolate(x_aspp, size=x_low.shape[-2:],
                               mode="bilinear", align_corners=False)
        x_low_p = self.low_project(x_low)

        x = torch.cat([x_aspp, x_low_p], dim=1)
        x = self.decoder(x)
        x = self.head(x)
        # Upsample bilinear final a la resolución de entrada
        return F.interpolate(x, size=in_size, mode="bilinear", align_corners=False)


# ╭───────────────────────────────────────────────────────────────────────╮
# │ Sanity check                                                          │
# ╰───────────────────────────────────────────────────────────────────────╯

if __name__ == "__main__":
    """Comprobación rápida sin GPU: forward de tamaño esperado para varios backbones."""
    NC, B, H, W = 21, 2, 256, 256
    for bb in ("resnet50",):                # con 50 vale para chequear; los demás van igual
        for os_ in (16, 8):
            print(f"[deeplab test] backbone={bb} output_stride={os_}")
            m = DeepLabV3Plus(num_classes=NC, backbone=bb, pretrained=False,
                              output_stride=os_, decoder_dropout=0.1)
            x = torch.randn(B, 3, H, W)
            y = m(x)
            assert y.shape == (B, NC, H, W), f"shape inesperada: {y.shape}"
            n_p = sum(p.numel() for p in m.parameters()) / 1e6
            print(f"    salida {tuple(y.shape)}, params {n_p:.1f} M  OK")
