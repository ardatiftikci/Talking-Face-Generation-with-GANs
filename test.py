import torch
import numpy as np
import pynvml
import torchvision
from PIL import Image
from pydub import AudioSegment
from torchvision import transforms
from identity_encoder import IdentityEncoder
from content_encoder import ContentRNN
from generator import Generator
from noise_generator import NoiseGenerator
from frame_discriminator import FrameDiscriminator
from utils import repeat_embeddings, cut_sequence
from torch.utils.data import DataLoader
from sequence_discriminator import SequenceDiscriminator
from tqdm import tqdm
from datasets import AudioDataset, FirstImageDataset, ImagesDataset


def save_video(index, video, audio):
    # torchvision.io.write_video(f"results/video{index}.mp4", video, fps=25, audio_array=audio, audio_fps=44100, audio_codec='mp3')
    torchvision.io.write_video(f"results/video{index}.mp4", video, fps=25)


def save_image(index, frame):
    frame = frame.detach().numpy().astype(np.uint8)
    im = Image.fromarray(frame)
    im.save(f"results/{index}.jpg")

# data read
BATCH_SIZE = 1
img_path = "../Talking-Face-Generation/data/LRW/ABOUT/train/ABOUT_{}.jpg"
imgs_path = "../Talking-Face-Generation/data/LRW/ABOUT/train/ABOUT_{}_{}.jpg"
audio_path = "../Talking-Face-Generation/data/LRW/ABOUT/train/ABOUT_{}.wav"

audio_dataset = AudioDataset(audio_path)
audio_loader = DataLoader(audio_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)
first_image_dataset = FirstImageDataset(img_path)
first_image_loader = DataLoader(first_image_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)
imgs_dataset = ImagesDataset(imgs_path)
images_loader = DataLoader(imgs_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)

# audio values
audio_sample_size = 8820
stride = int(44100 / float(25))
padding = audio_sample_size - stride

video_generator = Generator()
video_generator.load_state_dict(torch.load("./gen.pt"))
video_generator = video_generator.cuda()

for index, (audio_data, first_image_data, images_data) in enumerate(
        zip(audio_loader, first_image_loader, images_loader)):
    batch_size = audio_data.shape[0]
    audio_data_cut = cut_sequence(audio_data, stride, padding, audio_sample_size).cuda()
    audio_sequence_length = audio_data_cut.size()[1]
    first_image_data = first_image_data.cuda()
    fake_video = video_generator(batch_size, audio_data_cut, first_image_data, audio_sequence_length)
    video_to_save = fake_video.permute(0, 1, 3, 4, 2).squeeze(0).cpu()* 127.5 + 127.5
    save_video(index, video_to_save, audio_data.squeeze(2).cpu())
    save_image(index, video_to_save[0])
