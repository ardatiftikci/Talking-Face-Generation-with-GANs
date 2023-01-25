import os
import sys
import glob

import librosa
import librosa.display

import numpy as np

import torch
import torch.nn.functional as F
import torchvision as tv

import matplotlib.pyplot as plt

from PIL import Image
from IPython.display import Audio, display

from AudioCLIP.model import AudioCLIP
from AudioCLIP.utils.transforms import ToTensor1D
import pynvml


def get_memory_free_MiB():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(0))
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem_info.free // 1024 ** 2

class AudioClipEncoder():
    def __init__(self):
        self.MODEL_FILENAME = 'AudioCLIP-Full-Training.pt'
        self.IMAGE_SIZE = 224
        self.aclp = AudioCLIP(pretrained=f'AudioCLIP/assets/{self.MODEL_FILENAME}').cuda()
        self.aclp.requires_grad = False

    def __call__(self, x, z):
        print(get_memory_free_MiB())
        x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
        x = F.interpolate(x, size=self.IMAGE_SIZE)
        print(get_memory_free_MiB())
        z = z.reshape(-1, 1, z.shape[2])
        ((audio_features, _, _), _), _ = self.aclp(audio=z)
        print(get_memory_free_MiB())
        ((_, image_features, _), _), _ = self.aclp(image=x)
        print(get_memory_free_MiB())
        audio_features = audio_features / torch.linalg.norm(audio_features, dim=-1, keepdim=True)
        image_features = image_features / torch.linalg.norm(image_features, dim=-1, keepdim=True)
        scale_audio_image = torch.clamp(self.aclp.logit_scale_ai.exp(), min=1.0, max=100.0)
        logits_audio_image = torch.diagonal(scale_audio_image * audio_features @ image_features.T)
        return 1 - logits_audio_image / 100
