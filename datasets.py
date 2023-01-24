import numpy as np
from PIL import Image
from pydub import AudioSegment
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import audio
class AudioDataset(Dataset):
    def __init__(self, path):
        self.audio_max_value = 2 ** 15 - 1
        self.audios = np.array(
            [np.array(AudioSegment.from_file(path.format(str(i).zfill(5)), 'wav').set_channels(1)
                      .get_array_of_samples()).astype(np.int16) for i in range(1, 65)])

    def __getitem__(self, index):
        audio_data = torch.FloatTensor(self.audios[index] / self.audio_max_value).unsqueeze(1)
        return audio_data, torch.FloatTensor(audio.melspectrogram(audio_data)).squeeze(2).unsqueeze(0)

    def __len__(self):
        return len(self.audios)


class FirstImageDataset(Dataset):
    def __init__(self, path):
        self.img_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Resize((128, 128)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.first_images = torch.cat(([
            self.img_transform(np.array(Image.open(path.format(str(i).zfill(5))))).unsqueeze(0) for i in range(1, 65)]),
            dim=0)

    def __getitem__(self, index):
        return self.first_images[index]

    def __len__(self):
        return len(self.first_images)


class ImagesDataset(Dataset):
    def __init__(self, path):
        self.img_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Resize((128, 128)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.images = torch.stack([torch.cat(([
            self.img_transform(np.array(Image.open(path.format(str(i).zfill(5), j)))).unsqueeze(0) for j in range(30)]),
            dim=0) for i in range(1, 65)], dim=0)

    def __getitem__(self, index):
        return self.images[index]

    def __len__(self):
        return len(self.images)