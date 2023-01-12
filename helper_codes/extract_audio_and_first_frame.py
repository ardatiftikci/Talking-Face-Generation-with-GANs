import os
import moviepy.editor as mp
from PIL import Image
from tqdm import tqdm
path = 'data/LRW'
words = os.listdir(path)
for word in tqdm(words):
    for partition in ["test"]:
        videos_path = f"{path}/{word}/{partition}"
        videos = os.listdir(videos_path)
        for video in videos:
            if(video.endswith(".mp4")):
                video_path = f"{videos_path}/{video}"
                audio_path = video_path.replace(".mp4", ".mp3")
                #image_path = video_path.replace(".mp4", ".jpg")
                my_clip = mp.VideoFileClip(video_path)
                my_clip.audio.write_audiofile(audio_path,verbose=False, logger=None)
                #first_frame = my_clip.get_frame(0)
                #im = Image.fromarray(first_frame)
                #im.save(image_path)
