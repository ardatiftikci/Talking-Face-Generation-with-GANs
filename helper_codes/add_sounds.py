import os
import moviepy.editor as mp
from PIL import Image
from tqdm import tqdm
path = '../../Talking-Face-Generation/data/LRW'
word = "ABOUT"
partition = "train"
for j in range(2,15):
	for i in range(1,65):
		audio_path = f"{path}/{word}/{partition}/{word}_{str(i).zfill(5)}.wav"
		audio_background = mp.AudioFileClip(audio_path)
		my_clip = mp.VideoFileClip(f"../results_exp{j}/video{i-1}.mp4")
		final_clip = my_clip.set_audio(audio_background)
		final_clip.write_videofile(f"../results_exp{j}/video{i-1}.mp4", fps=25)
