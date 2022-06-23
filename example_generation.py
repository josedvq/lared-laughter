import os

import pandas as pd
from tqdm.notebook import tqdm

from lared.dataset.example import FullExample
from annotation.coco_reader import CocoReader

def make_folders(out_path):
    audio_path = os.path.join(out_path, 'audio')
    video_path = os.path.join(out_path, 'video')
    av_path = os.path.join(out_path, 'av')
    aiv_path = os.path.join(out_path, 'aiv')
    if not os.path.exists(audio_path): os.makedirs(audio_path)
    if not os.path.exists(video_path): os.makedirs(video_path)
    if not os.path.exists(av_path): os.makedirs(av_path)
    if not os.path.exists(aiv_path): os.makedirs(aiv_path)


def generate_media(cvat_path, examples_path, out_path, img_path = './audio.jpg'):
    audio_path = os.path.join(out_path, 'audio')
    video_path = os.path.join(out_path, 'video')
    av_path = os.path.join(out_path, 'av')
    aiv_path = os.path.join(out_path, 'aiv')

    coco_reader = CocoReader(cvat_path)

    examples_df = pd.read_csv(examples_path, index_col=0)
    examples = {
        ex[1].hash: FullExample(**ex[1].to_dict()) for ex in examples_df.iterrows()
    }

    # write back the bbs
    for hash, ex in tqdm(examples.items()):
        if hash not in coco_reader:
            continue
        if len(coco_reader[hash]) != 4:
            continue
        ex.rect = coco_reader[hash]
        ex.write_audio(audio_path, padded=True)
        ex.write_audio_video(audio_path, img_path, aiv_path, padded=True)
        ex.write_video(video_path, padded=True)
        ex.write_audiovisual(video_path, audio_path, av_path)

    return examples