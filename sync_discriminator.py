import torch
import torch.nn as nn
from identity_encoder import IdentityEncoder
from content_encoder import ContentRNN


class SyncDiscriminator(nn.Module):
    def __init__(self, batch_size):
        super(SyncDiscriminator, self).__init__()
        self.batch_size = batch_size
        # image encoder
        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(5, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
            )

        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(inplace=True)
                                   )
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(inplace=True)
                                   )
        self.conv4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(inplace=True)
                                   )

        self.image_linear = nn.Linear(512 * 4 * 8, 256, bias=False)

        # audio encoder

        self.conv5 = nn.Sequential(nn.Conv1d(1, 16, 250, stride=50, padding=700),
                                   nn.BatchNorm1d(16),
                                   nn.ReLU(inplace=True))

        self.conv6 = nn.Sequential(nn.Conv1d(16, 32, 4, stride=2, padding=1),
                                   nn.BatchNorm1d(32),
                                   nn.ReLU(inplace=True))

        self.conv7 = nn.Sequential(nn.Conv1d(32, 64, 4, stride=2, padding=1),
                                   nn.BatchNorm1d(64),
                                   nn.ReLU(inplace=True))

        self.conv8 = nn.Sequential(nn.Conv1d(64, 128, 4, stride=2, padding=1),
                                   nn.BatchNorm1d(128),
                                   nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(nn.Conv1d(128, 256, 10, stride=5, padding=4),
                                   nn.BatchNorm1d(256),
                                   nn.ReLU(inplace=True))

        self.audio_linear = nn.Linear(256 * 5, 256, bias=False)

        self.audio_bilstm = nn.LSTM(batch_first=True, input_size=256, hidden_size=256, bidirectional=True)
        self.image_bilstm = nn.LSTM(batch_first=True, input_size=256, hidden_size=256, bidirectional=True)
        self.audio_hidden = self.init_hidden(128)
        self.image_hidden = self.init_hidden(128)
        self.slp = nn.Linear(1, 1, bias=True)
        self.activation = nn.Sigmoid()

    def init_hidden(self, hidden_dim):
        return (torch.zeros(2, self.batch_size, hidden_dim).cuda(),
                torch.zeros(2, self.batch_size, hidden_dim).cuda())

    def forward(self, x, z):
        x = torch.permute(x, (0, 2, 1, 3, 4))
        x = self.conv1(x).squeeze(2)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x).reshape(self.batch_size, -1)
        x = self.image_linear(x).unsqueeze(1)
        image_out, self.image_hidden = self.image_bilstm(x)
        image_out = image_out.squeeze(1)

        z = torch.permute(z, (0, 2, 1))
        z = self.conv5(z)
        z = self.conv6(z)
        z = self.conv7(z)
        z = self.conv8(z)
        z = self.conv9(z).reshape(self.batch_size, -1)
        z = self.audio_linear(z).unsqueeze(1)
        audio_out, self.audio_hidden = self.audio_bilstm(z)
        audio_out = audio_out.squeeze(1)
        dist = torch.diagonal(torch.cdist(image_out, audio_out)).view(-1, 1)
        out = self.slp(dist)

        self.audio_hidden = (self.audio_hidden[0].detach(), self.audio_hidden[1].detach())
        self.image_hidden = (self.image_hidden[0].detach(), self.image_hidden[1].detach())
        return self.activation(out)
