import os
import subprocess

import cv2

import numpy as np
from scipy.stats import norm
from scipy.io import wavfile

def make_audio(fout_path, fps, on_intervals, f=48000):
    length = int(on_intervals[-1][1] * f / fps)
    Y = np.zeros((length))

    for on_int in on_intervals:
        ini_time = on_int[0] / fps
        end_time = on_int[1] / fps

        t = np.linspace(ini_time, end_time, int(f * (end_time - ini_time)))
        y = np.sin(1000 * 2 * np.pi * t)  #  Has frequency of 440Hz
        Y[int(ini_time * f): int(ini_time * f) + len(y)] = y

    wavfile.write(fout_path, f, Y)

def make_video(fout_path, fps, on_intervals, res=(1920, 1080)):
    # generate segment video
    fout = cv2.VideoWriter(fout_path, cv2.VideoWriter_fourcc(*'H264'), fps, res)

    off_intervals = [[0, on_intervals[0][0]]]
    for i in range(len(on_intervals)-1):
        off_intervals.append([on_intervals[i][1], on_intervals[i+1][0]])

    for on_int, off_int in zip(on_intervals, off_intervals):
        for i in range(off_int[0], off_int[1]):
            blank_frame = np.zeros((res[1],res[0],3), np.uint8)

            fout.write(blank_frame)

        for i in range(on_int[0], on_int[1]):
            frame = np.zeros((res[1],res[0],3), np.uint8)
            frame = cv2.circle(frame, center=(res[0]//2, res[1]//2), radius = res[1]//8, color=(255,255,255), thickness=-1)
            fout.write(frame)
    
    fout.release()

def make_audiovisual(fbasename, num_on, fps, on_distribution, off_distribution):

    intervals = []
    time_intervals = []
    prev_frame = 1+int(off_distribution.rvs() * fps) # wait at least one second
    for i in range(num_on):
        curr_frame = prev_frame + int(on_distribution.rvs() * fps)
        intervals.append([prev_frame, curr_frame])
        time_intervals.append([prev_frame / fps, curr_frame / fps])

        prev_frame = curr_frame + int(off_distribution.rvs() * fps)

    make_video(f'{fbasename}_video.mp4', fps, intervals)
    make_audio(f'{fbasename}_audio.wav', fps, intervals)

    av_path = f'{fbasename}_av.mp4'
    cmd = ['ffmpeg', '-i', f'{fbasename}_video.mp4', '-i', f'{fbasename}_audio.wav', '-c:v copy', '-c:a aac', av_path]
    cmdstr = ' '.join(cmd)

    if os.path.isfile(av_path):
        os.remove(av_path)

    process = subprocess.run(cmdstr,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                shell=True,
                                check=False)

    return time_intervals