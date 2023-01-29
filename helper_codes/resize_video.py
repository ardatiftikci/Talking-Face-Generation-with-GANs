import os
import moviepy.editor as mp
from PIL import Image
from tqdm import tqdm
import soundfile as sf
from mutagen.mp3 import MP3
path = '../results_comparison'
final_path = '../results_comparison_resized'
for i in range(1,65):
    video_path = f"ABOUT_{str(i).zfill(5)}.mp4"
    my_clip = mp.VideoFileClip(f"{path}/{video_path}")
    final_clip = my_clip.resize((128,128))
    final_clip.write_videofile(f"{final_path}/video{i-1}.mp4", fps=25)
