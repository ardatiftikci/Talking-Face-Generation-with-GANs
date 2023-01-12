import numpy as np
import torch.nn as nn
import torch
from utils import repeat_embeddings

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels + skip_channels, in_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, s):
        s = s.view(x.shape)
        x = torch.cat([x, s], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x).view(-1, x.shape[1]//2, x.shape[2]*2, x.shape[3]*2)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class Generator(nn.Module):
    def __init__(self, img_size=(128, 128), audio_latent_size=128, identity_latent_size=128, noise_size=10, skip_channels=[]):
        super(Generator, self).__init__()
        self.img_size = img_size

        self.audio_latent_size = audio_latent_size
        self.identity_latent_size = identity_latent_size
        self.noise_size = noise_size
        self.total_latent_size = self.audio_latent_size + self.identity_latent_size + self.noise_size

        input_size = (4, 4)
        channels = 1024

        self.layers = nn.ModuleList()

        first_layer = nn.Sequential(
            nn.ConvTranspose2d(self.total_latent_size, channels, input_size, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))

        self.layers.append(first_layer)
        for i in range(4):
            self.layers.append(
                UNetBlock(channels, channels // 2, skip_channels[i]))

            channels //= 2
            input_size = tuple(2 * x for x in input_size)

        self.layers.append(
            nn.ConvTranspose2d(channels, 3, 4, stride=2, padding=1, bias=False))
        self.activation = nn.Tanh()

    def forward(self, audio_latent, identity_latent, noise, skip_connections):
        batch_size = audio_latent.shape[0]
        x = torch.cat([repeat_embeddings(identity_latent, audio_latent.shape[1]).cuda(), audio_latent, noise], dim=2)
        x = x.view(-1, self.total_latent_size, 1, 1)
        x = self.layers[0](x)
        for i in range(1, 5):
            x = self.layers[i](x, repeat_embeddings(skip_connections[i - 1], audio_latent.shape[1]).cuda())
        x = self.layers[-1](x, output_size=[-1, 3, self.img_size[0], self.img_size[1]])
        return self.activation(x).view(batch_size, -1, 3, self.img_size[0], self.img_size[1])
