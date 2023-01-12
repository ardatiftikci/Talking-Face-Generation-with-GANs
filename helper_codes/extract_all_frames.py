import os
import moviepy.editor as mp
from PIL import Image
from tqdm import tqdm
path = 'data/LRW'
word = "ABOUT"
partition = "train"
for i in range(1,65):
    video_path = f"{path}/{word}/{partition}/{word}_{str(i).zfill(5)}.mp4"
    my_clip = mp.VideoFileClip(video_path)
    for j in range(30):
    	frame = my_clip.get_frame(j)
    	im = Image.fromarray(frame)
    	im.save(f"{path}/{word}/{partition}/{word}_{str(i).zfill(5)}_{j}.jpg")
