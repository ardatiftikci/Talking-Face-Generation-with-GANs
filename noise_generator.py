import torch
import torch.nn as nn
import math


class NoiseGenerator(nn.Module):
    def __init__(self):
        super(NoiseGenerator, self).__init__()
        self.std = math.sqrt(0.6)
        self.gru = nn.GRU(10, 10, batch_first=True)
        self.activation = nn.Tanh()

    def forward(self, batch_size, audio_length):
        noise = torch.randn((batch_size, audio_length, 10)).cuda() * self.std
        noise, h = self.gru(noise)
        return self.activation(noise)
