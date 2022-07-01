import os
import pickle
import logging
logger = logging.getLogger(__name__)
handler = logging.FileHandler('extractor.log')
handler.setLevel(logging.INFO)
logger.addHandler(handler)

from joblib import Parallel, delayed
import torch
import numpy as np
from pytorchvideo.data.clip_sampling import ClipSampler
from pytorchvideo.data.video import VideoPathHandler
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
from jose.helpers.timing import timing

class VideoExtractor():
    _MAX_CONSECUTIVE_FAILURES = 10

    def __init__(
        self,
        videos_path: str,
        transform: Optional[Callable[[dict], Any]] = None,
        sr = 30, # fps
        n_jobs = 1
    ) -> None:
        self.videos_path = videos_path
        self.transform = transform

        # If a RandomSampler is used we need to pass in a custom random generator that
        # ensures all PyTorch multiprocess workers have the same random seed.
        self._video_random_generator = None
        self.video_path_handler = VideoPathHandler()

        self.sr = sr
        self.n_jobs = n_jobs
        self.cache = {}

    def extract_multiple(self, keys):
        return np.stack(Parallel(n_jobs=self.n_jobs)(delayed(self.__call__)(*k) for k in keys))
        return np.stack([self(*k) for k in keys])

    def _get_clip(self, key, start, end):
        if key in self.cache:
            video = self.cache[key]
        else:
            video = self.video_path_handler.video_from_path(
                os.path.join(self.videos_path, f'{key}.mp4'),
                decoder='pyav'
            )

            video = video.get_clip(0, float(video.duration))

            video_is_null = (video is None or video["video"] is None)
            
            if video_is_null:
                raise Exception(f"Failed to load clip {video.name}, for start={start}, end={end}")

            video = video['video'].to(torch.uint8)
            self.cache[key] = video
            
        start_frame = round(start * self.sr)
        end_frame = round(end * self.sr)

        frames = video[:,start_frame:end_frame,:,:].to(torch.float32)

        if self.transform is not None:
            frames = self.transform(frames)

        return frames

    def __call__(self, key, start, end) -> dict:
        # logger.info((key, start, end))

        return self._get_clip(key, start, end)

        for i_try in range(self._MAX_CONSECUTIVE_FAILURES):
            
            video = self.video_path_handler.video_from_path(
                os.path.join(self.videos_path, f'{key}.mp4'),
                decoder='pyav'
            )

            loaded_clip = video.get_clip(0, video._duration)
            
            video_is_null = (loaded_clip is None or loaded_clip["video"] is None)
            
            if video_is_null:
                raise Exception(f"Failed to load clip {video.name}, for start={start}, end={end}")

            frames = loaded_clip["video"]


            if self.transform is not None:
                frames = self.transform(frames)

            return frames
        else:
            raise RuntimeError(
                f"Failed to load video {key} after {self._MAX_CONSECUTIVE_FAILURES} retries."
            )
