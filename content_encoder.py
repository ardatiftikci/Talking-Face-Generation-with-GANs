import torch
import torch.nn as nn
import math


class ContentEncoder(nn.Module):
    def __init__(self, encoding_size):
        super(ContentEncoder, self).__init__()

        self.encoding_size = encoding_size
        self.activations = nn.ModuleList()
        self.strides = [50, 2, 2, 2, 5, 5]
        self.kernels = [250, 4, 4, 4, 10, 10]
        self.paddings = [230, 3, 3, 3, 7]
        self.layers = nn.ModuleList()
        self.channels = []
        for i in range(5):
            pad = int(math.ceil(self.paddings[i] / 2.0))
            in_channels = self.channels[-1] if len(self.channels) > 0 else 1
            out_channels = 2 * self.channels[-1] if len(self.channels) > 0 else 16

            self.layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, self.kernels[i],
                          stride=self.strides[i], padding=pad),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(True)))
            self.channels.append(out_channels)

        self.layers.append(nn.Sequential(
            nn.Conv1d(self.channels[-1], self.encoding_size, 5),
            nn.Tanh()))

    def forward(self, x):
        for i in range(len(self.strides)):
            x = self.layers[i](x)
        return x.squeeze()


class ContentRNN(nn.Module):
    def __init__(self, encoding_size=128, gru_hidden_size=128):
        super(ContentRNN, self).__init__()
        self.sample_count = int(44100 * 0.2)
        self.encoding_size = encoding_size
        self.gru_hidden_size = gru_hidden_size
        self.content_encoder = ContentEncoder(self.encoding_size)
        self.rnn = nn.GRU(self.encoding_size, self.gru_hidden_size, 1, batch_first=True)

    def forward(self, x):
        audio_length = x.shape[1]
        x = x.view(-1, 1, self.sample_count)
        x = self.content_encoder(x)
        x = x.view(-1, audio_length, self.encoding_size)
        x, h = self.rnn(x)
        return x
