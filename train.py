import pynvml
import torch
from generator import Generator
from frame_discriminator import FrameDiscriminator
from utils import cut_sequence
from torch.utils.data import DataLoader
from sequence_discriminator import SequenceDiscriminator
from tqdm import tqdm
from datasets import AudioDataset, FirstImageDataset, ImagesDataset
from syncnet import SyncNet_color
from pose_info_extraction import PoseInfoExtraction
import numpy as np
def synergynetLoss(SnetModels,x):
  model, face_boxes = SnetModels


def get_memory_free_MiB():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(0))
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem_info.free // 1024 ** 2

logloss = torch.nn.BCELoss()
def cosine_loss(a, v, y):
    d = torch.nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)
    return loss
def get_sync_loss(syncnet, mel, g):
    g = g[:, :, g.size(3)//2:,:]
    g = torch.cat([g for i in range(5)], dim=1)
    a, v = syncnet(torch.randn((30,1,441, 441)).cuda(), g)
    y = torch.ones(g.size(0), 1).float().cuda()
    return cosine_loss(a, v, y)

# data read
BATCH_SIZE = 1
img_path = "../lrw_subset/ABOUT_{}.jpg"
imgs_path = "../lrw_subset/ABOUT_{}_{}.jpg"
audio_path = "../lrw_subset/ABOUT_{}.wav"

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

from collections import deque

deq = deque(maxlen=2)
deq.append(get_memory_free_MiB())

video_generator = Generator().cuda()
deq.append(get_memory_free_MiB())
print(f"generator {deq[0] - deq[1]} MB")

d_frame = FrameDiscriminator().cuda()
deq.append(get_memory_free_MiB())
print(f"frame disc {deq[0] - deq[1]} MB")

d_seq = SequenceDiscriminator().cuda()
deq.append(get_memory_free_MiB())

print(get_memory_free_MiB())
#syncnet = SyncNet_color().cuda()
print(get_memory_free_MiB())
g_optimizer = torch.optim.Adam(video_generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
d_frame_optimizer = torch.optim.Adam(d_frame.parameters(), lr=1e-4, betas=(0.5, 0.999))
d_seq_optimizer = torch.optim.Adam(d_seq.parameters(), lr=1e-5, betas=(0.5, 0.999))

pose_extractor = PoseInfoExtraction()
mse_loss = torch.nn.MSELoss()
l1_loss = torch.nn.L1Loss()
pbar = tqdm(range(300))
for epoch in pbar:
    for index, (audio_data, first_image_data, images_data) in enumerate(
            zip(audio_loader, first_image_loader, images_loader)):
        batch_size = audio_data.shape[0]

        audio_data = cut_sequence(audio_data, stride, padding, audio_sample_size).cuda()
        audio_sequence_length = audio_data.size()[1]
        first_image_data = first_image_data.cuda()
        images_data = images_data.cuda()

        real_labels = torch.ones(batch_size * audio_sequence_length).cuda()
        fake_labels = torch.zeros(batch_size * audio_sequence_length).cuda()

        for i in range(1):
            # Generate video for discriminators
            fake_video = video_generator(batch_size, audio_data, first_image_data, audio_sequence_length)

            #x = get_sync_loss(syncnet, audio_data.squeeze(0).unsqueeze(1), fake_video.squeeze(0))
            #print(x)

            # Frame Discriminator
            outputs = d_frame(images_data, first_image_data).view(-1)
            d_frame_loss_real = mse_loss(outputs, real_labels)

            outputs = d_frame(fake_video, first_image_data).view(-1)
            d_frame_loss_fake = mse_loss(outputs, fake_labels)
            d_frame_optimizer.zero_grad()

            d_frame_loss = d_frame_loss_real + d_frame_loss_fake
            d_frame_loss.backward(retain_graph=True)
            d_frame_optimizer.step()

            # Sequence Discriminator
            outputs = d_seq(images_data, audio_data).view(-1)
            d_seq_loss_real = mse_loss(outputs, real_labels)

            outputs = d_seq(fake_video, audio_data).view(-1)
            d_seq_loss_fake = mse_loss(outputs, fake_labels)
            d_seq_optimizer.zero_grad()

            d_seq_loss = d_seq_loss_real + d_seq_loss_fake
            d_seq_loss.backward()
            d_seq_optimizer.step()


        # Generator Loss
        for _ in range(1):
            fake_video = video_generator(batch_size, audio_data, first_image_data, audio_sequence_length)

            outputs_frame = d_frame(fake_video, first_image_data).view(-1)
            outputs_seq = d_seq(fake_video, audio_data).view(-1)

            lower_face_recons_loss = torch.mean(torch.abs(fake_video - images_data)[:, :, :, 64:, :])

            pose_extraction = pose_extractor(fake_video).view(-1, 62)
            pose_loss = torch.mean(torch.abs(pose_extraction[1:, :] - pose_extraction[:-1, :]))

            g_loss = mse_loss(outputs_frame, real_labels) + mse_loss(outputs_seq, real_labels)
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

    pbar.set_description(
        (
            f"epoch: {epoch}\n"
            f"d_frame_loss: {d_frame_loss.item():.4f}\n"
            f"d_seq_loss: {d_seq_loss.item():.4f}\n"
            f"g_loss: {g_loss.item():.4f}\n"
        )
    )
    if i % 20 == 19:
        torch.save(video_generator.state_dict(), "./gen2.pt")
        torch.save(d_frame.state_dict(), "./d_frame2.pt")
        torch.save(d_seq.state_dict(), "./d_seq2.pt")
