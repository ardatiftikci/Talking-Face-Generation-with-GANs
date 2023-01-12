import torch
import torch.nn as nn
from content_encoder import ContentRNN


class SequenceDiscriminator(nn.Module):
    def __init__(self):
        super(SequenceDiscriminator, self).__init__()
        self.audio_encoder = ContentRNN()
        """
        self.channels = []
        self.layers = nn.ModuleList()
        # 6 layers CNN mentioned in the paper
        for i in range(5):
            in_channels = self.channels[-1] if len(self.channels) > 0 else 3
            out_channels = 2 * self.channels[-1] if len(self.channels) > 0 else 16
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, stride=1, padding=2, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)))
            self.channels.append(out_channels)

        self.layers.append(nn.Sequential(nn.Conv2d(out_channels, 128, kernel_size=6, stride=1, bias=False),
                                         nn.BatchNorm2d(128),
                                         nn.ReLU(inplace=True)))

        self.fc = nn.Linear(320, 1, bias=False)
        self.Gru = nn.GRU(256, 64, 2, batch_first=True)
        self.sig = nn.Sigmoid()
        """
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=(4, 4), padding=(1, 1), stride=(2, 2), bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(0.1, inplace=True)
                                   )
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=(4, 4), padding=(1, 1), stride=(2, 2), bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(0.1, inplace=True)
                                   )
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=(4, 4), padding=(1, 1), stride=(2, 2), bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(0.1, inplace=True)
                                   )
        self.conv4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=(4, 4), padding=(1, 1), stride=(2, 2), bias=False),
                                   nn.BatchNorm2d(512),
                                   nn.LeakyReLU(0.1, inplace=True)
                                   )
        self.conv5 = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=(4, 4), padding=(1, 1), stride=(2, 2), bias=False),
                                   nn.BatchNorm2d(1024),
                                   nn.LeakyReLU(0.1, inplace=True)
                                   )
        self.conv6 = nn.Sequential(nn.Conv2d(1024, 128, kernel_size=4, stride=1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(0.1, inplace=True)
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
        z = self.audio_encoder(z).squeeze(0)
        x = x.view(x.shape[0], -1)
        x = torch.cat((x, z), dim=1)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.activation(x)
        return x.view(batch_size, -1)
