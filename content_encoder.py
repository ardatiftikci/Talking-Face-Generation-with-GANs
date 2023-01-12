import torch
import torch.nn as nn
import math


class ContentEncoder(nn.Module):
    def __init__(self, encoding_size):
        super(ContentEncoder, self).__init__()

        self.encoding_size = encoding_size
        self.conv1 = nn.Sequential(nn.Conv1d(1, 16, 250,
                                             stride=50, padding=115),
                                   nn.BatchNorm1d(16),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(nn.Conv1d(16, 32, 4,
                                             stride=2, padding=2),
                                   nn.BatchNorm1d(32),
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(nn.Conv1d(32, 64, 4,
                                             stride=2, padding=2),
                                   nn.BatchNorm1d(64),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(nn.Conv1d(64, 128, 4,
                                             stride=2, padding=2),
                                   nn.BatchNorm1d(128),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(nn.Conv1d(128, 256, 10,
                                             stride=5, padding=4),
                                   nn.BatchNorm1d(256),
                                   nn.ReLU(inplace=True))

        self.conv6 = nn.Sequential(nn.Conv1d(256, self.encoding_size, 5),
                                   nn.Tanh())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
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
