import torch
import torch.nn as nn


class IdentityEncoder(nn.Module):
    def __init__(self, img_size=(128, 128), encoding_size=128):
        super(IdentityEncoder, self).__init__()
        self.img_size = img_size
        self.code_size = encoding_size
        self.layers = nn.ModuleList()
        self.channels = []

        # 6 layers CNN mentioned in the paper
        for i in range(5):
            in_channels = self.channels[-1] if len(self.channels) > 0 else 3
            out_channels = 2 * self.channels[-1] if len(self.channels) > 0 else 64
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)))
            self.channels.append(out_channels)

        # final layer
        self.layers.append(nn.Sequential(
            nn.Conv2d(self.channels[-1], encoding_size, 4, bias=False),
            nn.Tanh()))

    def forward(self, x, skip_connections=False):
        if skip_connections:
            skips = [x]
            for layer in self.layers:
                skips.append(layer(skips[-1]))
            out = skips[-1].view(-1, self.code_size)
            skips_internal = skips[1:-1]
            return out, skips_internal
        else:
            for layer in self.layers:
                x = layer(x)
            out = x.view(-1, self.code_size)
            return out
