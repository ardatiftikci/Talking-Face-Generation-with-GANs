import torch
import torch.nn as nn


class FrameDiscriminator(nn.Module):
    def __init__(self):
        super(FrameDiscriminator, self).__init__()
        self.channels = []
        self.layers = nn.ModuleList()
        # 6 layers CNN mentioned in the paper
        for i in range(5):
            in_channels = self.channels[-1] if len(self.channels) > 0 else 6
            out_channels = 2 * self.channels[-1] if len(self.channels) > 0 else 64
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)))
            self.channels.append(out_channels)

        self.layers.append(nn.Conv2d(out_channels, 1, kernel_size=4, stride=1, bias=False))

        self.activation = nn.Sigmoid()

    def forward(self, x, c):
        batch_size = x.shape[0]
        x = torch.cat((x, c.unsqueeze(1).repeat(1, 30, 1, 1, 1)), dim=2).view(-1, x.shape[2] * 2, x.shape[3], x.shape[4])
        for layer in self.layers:
            x = layer(x)
        x = x.view(batch_size, -1)
        x = self.activation(x)
        return x
