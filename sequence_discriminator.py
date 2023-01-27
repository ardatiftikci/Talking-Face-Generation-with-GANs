import torch
import torch.nn as nn
from content_encoder import ContentRNN


class SequenceDiscriminator(nn.Module):
    def __init__(self):
        super(SequenceDiscriminator, self).__init__()
        self.audio_encoder = ContentRNN()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=4, padding=1, stride=2, bias=False),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU(inplace=True)
                                   )
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=4, padding=1, stride=2, bias=False),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(inplace=True)
                                   )
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=4, padding=1, stride=2, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True)
                                   )
        self.conv4 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=4, padding=1, stride=2, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(inplace=True)
                                   )
        self.conv5 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=4, padding=1, stride=2, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(inplace=True)
                                   )
        self.conv6 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=4, stride=1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(inplace=True)
                                   )
        self.gru = nn.GRU(128, 32, 1, batch_first=True)
        self.linear = nn.Linear(160, 1, bias=False)
        self.activation = nn.Sigmoid()

    def forward(self, x, z):
        batch_size = x.shape[0]
        x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.shape[0], 1, -1)
        x, _ = self.gru(x)
        x = x.view(batch_size, 30, 32)
        z = self.audio_encoder(z).squeeze(0)
        x = torch.cat((x, z), dim=2)
        x = x.view(-1, 160)
        x = self.linear(x)
        x = self.activation(x)
        return x
