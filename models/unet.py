import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = models.resnet50(weights=weights)
        self.layer0 = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1   # /4 ,  256 ch
        self.layer2 = net.layer2   # /8 ,  512 ch
        self.layer3 = net.layer3   # /16, 1024 ch
        self.layer4 = net.layer4   # /32, 2048 ch

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x4, [x3, x2, x1, x0]


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class UNet(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.encoder = Encoder(pretrained=pretrained)
        self.dec1 = DecoderBlock(2048, 1024, 512)   # 7  -> 14
        self.dec2 = DecoderBlock( 512,  512, 256)   # 14 -> 28
        self.dec3 = DecoderBlock( 256,  256, 128)   # 28 -> 56
        self.dec4 = DecoderBlock( 128,   64,  64)   # 56 -> 112
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        in_size = x.shape[-2:]
        bottleneck, skips = self.encoder(x)          # skips = [x3, x2, x1, x0]
        d = self.dec1(bottleneck, skips[0])
        d = self.dec2(d,          skips[1])
        d = self.dec3(d,          skips[2])
        d = self.dec4(d,          skips[3])
        out = self.head(d)
        return F.interpolate(out, size=in_size, mode="bilinear", align_corners=False)
