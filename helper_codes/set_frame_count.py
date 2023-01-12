import os
import moviepy.editor as mp
from PIL import Image
from tqdm import tqdm
import soundfile as sf
from mutagen.mp3 import MP3
path = '../MakeItTalk_results'
second_path = 'data/LRW'
final_path = '../MakeItTalk_results2'
words = os.listdir(path)
for word in tqdm(words):
    try:
        os.mkdir(f"{final_path}/{word}")
    except:
        pass
    for partition in ["test"]:
        videos_path = f"{path}/{word}"
        videos = os.listdir(videos_path)
        for video in tqdm(videos):
            if video.endswith(".mp4"):
                filename = video[:-14]
                my_clip = mp.VideoFileClip(f"{videos_path}/{video}")
                audio_background = mp.AudioFileClip(f"{second_path}/{word}/{partition}/{filename}.mp3")
                my_clip = my_clip.set_duration(audio_background.duration)
                final_clip = my_clip.set_audio(audio_background)
                final_clip.write_videofile(f"{final_path}/{word}/{filename}_generated.mp4", fps=25)