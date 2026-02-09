import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, use_bn=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, k, s, p, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FeatureExtractor(nn.Module):
    """Simple CNN encoder used for feature extraction."""

    def __init__(self, in_ch=3, base=64):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_ch, base),
            ConvBlock(base, base),
            nn.MaxPool2d(2),
            ConvBlock(base, base * 2),
            ConvBlock(base * 2, base * 2),
            nn.MaxPool2d(2),
            ConvBlock(base * 2, base * 4),
            ConvBlock(base * 4, base * 4),
        )

    def forward(self, x):
        return self.encoder(x)


class Generator(nn.Module):
    def __init__(self, in_ch=3, base=64):
        super().__init__()
        self.encoder = FeatureExtractor(in_ch, base)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, in_ch, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        features = self.encoder(x)
        out = self.decoder(features)
        return out


class Discriminator(nn.Module):
    """PatchGAN discriminator for image realism."""

    def __init__(self, in_ch=3, base=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base * 2, 4, 2, 1),
            nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1),
            nn.BatchNorm2d(base * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 4, 1, 4, 1, 1),
        )

    def forward(self, x):
        return self.net(x)
