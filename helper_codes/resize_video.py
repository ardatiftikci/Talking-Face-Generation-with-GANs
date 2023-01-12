import os
import moviepy.editor as mp
from PIL import Image
from tqdm import tqdm
import soundfile as sf
from mutagen.mp3 import MP3
path = 'SDA_results'
final_path = 'SDA_results_resized'
words = os.listdir(path)
for word in tqdm(words):
    try:
        os.mkdir(f"{final_path}/{word}")
    except:
        pass
    videos_path = f"{path}/{word}"
    videos = os.listdir(videos_path)
    for video in videos:
        filename = video[:-14]
        my_clip = mp.VideoFileClip(f"{videos_path}/{video}")
        final_clip = my_clip.resize((256,256))
        final_clip.write_videofile(f"{final_path}/{word}/{filename}_generated.mp4", fps=25)
