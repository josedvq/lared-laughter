import os
from hashlib import sha256
import subprocess

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.io import wavfile
import cv2

class AmiExample:
    corpus_path = None

    @staticmethod
    def get_video_cap(meeting, participant):
        return cv2.VideoCapture(os.path.join(
            AmiExample.corpus_path, 
            f'{meeting}/video/{meeting}.Closeup{participant}.avi'))

    @staticmethod
    def get_audio(meeting, participant):
        fpath = os.path.join(
            AmiExample.corpus_path, 
            f'{meeting}/audio/{meeting}.Headset-{participant-1}.wav')
        return wavfile.read(fpath)

    def __init__(self, meeting: str, participant: int, ini: float, end: float):
        self.meeting = meeting
        self.participant = participant
        self.ini = ini
        self.end = end

    def get_example_id(self):
        return f'{self.meeting}_{self.participant:d}_{self.ini:.02f}_{self.end:.02f}'

    def get_hash_id(self):
        return sha256(self.get_example_id().encode()).hexdigest()

    def write_video(self, fout_path):
        fout_path = os.path.join(fout_path, f'{self.get_hash_id():s}.mp4')

        vcap = self.get_video_cap(self.meeting, self.participant)
        fps = vcap.get(cv2.CAP_PROP_FPS)
        w = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video_ini = int(self.ini * fps)
        video_len = int((self.end-self.ini) * fps)

        # generate segment video
        fout = cv2.VideoWriter(fout_path, cv2.VideoWriter_fourcc(*'H264'), fps, (w,h))
        vcap.set(cv2.CAP_PROP_POS_FRAMES, video_ini)

        curr_frame = 0
        while curr_frame < video_len:
            _, frame = vcap.read()
            fout.write(frame)
            curr_frame += 1

        fout.release()

    def write_audio(self, fout_path):
        fout_path = os.path.join(fout_path, f'{self.get_hash_id():s}.wav')
        fs, data = AmiExample.get_audio(self.meeting, self.participant)

        episode = data[fs * self.ini: fs * self.end]
        wavfile.write(fout_path, fs, episode)

    def write_audiovisual(self, video_path, audio_path, fout_path):

        video_path = os.path.join(video_path, f'{self.get_hash_id():s}.mp4')

        audio_path = os.path.join(audio_path, f'{self.get_hash_id():s}.wav')

        av_path = os.path.join(fout_path, f'{self.get_hash_id():s}-av.mp4')

        cmd = ['ffmpeg', '-i', video_path, '-i',
               audio_path, '-c:v copy', '-c:a aac', av_path]
        cmdstr = ' '.join(cmd)

        process = subprocess.run(cmdstr,
                                 stdout=subprocess.PIPE,
                                 shell=True,
                                 check=False)
