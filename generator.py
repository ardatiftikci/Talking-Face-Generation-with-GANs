import numpy as np
import torch.nn as nn
import torch
from utils import repeat_embeddings
from identity_encoder import IdentityEncoder
from content_encoder import ContentRNN
from noise_generator import NoiseGenerator

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
        self.content_encoder = ContentRNN()
        self.id_encoder = IdentityEncoder()
        self.noise_gen = NoiseGenerator()

        self.skip_channels = list(self.id_encoder.channels)
        self.skip_channels.reverse()

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
                UNetBlock(channels, channels // 2, self.skip_channels[i]))

            channels //= 2
            input_size = tuple(2 * x for x in input_size)

        self.layers.append(
            nn.ConvTranspose2d(channels, 3, 4, stride=2, padding=1, bias=False))
        self.activation = nn.Tanh()

    def forward(self, batch_size, audio_data, first_image_data, audio_sequence_length):
        audio_latent = self.content_encoder(audio_data)
        identity_latent, identity_skips = self.id_encoder(first_image_data, skip_connections=True)
        skip_connections = []
        for skip_variable in identity_skips:
            skip_connections.append(skip_variable.cuda())
        skip_connections.reverse()
        noise = self.noise_gen(batch_size, audio_sequence_length).cuda()

        batch_size = audio_latent.shape[0]
        x = torch.cat([repeat_embeddings(identity_latent, audio_latent.shape[1]).cuda(), audio_latent, noise], dim=2)
        x = x.view(-1, self.total_latent_size, 1, 1)
        x = self.layers[0](x)
        for i in range(1, 5):
            x = self.layers[i](x, repeat_embeddings(skip_connections[i - 1], audio_latent.shape[1]).cuda())
        x = self.layers[-1](x, output_size=[-1, 3, self.img_size[0], self.img_size[1]])
        return self.activation(x).view(batch_size, -1, 3, self.img_size[0], self.img_size[1])
