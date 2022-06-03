import os

from pytorchvideo.data.clip_sampling import ClipSampler
from pytorchvideo.data.video import VideoPathHandler
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

class VideoExtractor():
    _MAX_CONSECUTIVE_FAILURES = 10

    def __init__(
        self,
        videos_path: str,
        clip_sampler: ClipSampler,
        transform: Optional[Callable[[dict], Any]] = None,
    ) -> None:
        self.videos_path = videos_path
        self.transform = transform
        self.clip_sampler = clip_sampler

        # If a RandomSampler is used we need to pass in a custom random generator that
        # ensures all PyTorch multiprocess workers have the same random seed.
        self._video_random_generator = None
        self.video_path_handler = VideoPathHandler()

    def __getitem__(self, key, start, end) -> dict:
        for i_try in range(self._MAX_CONSECUTIVE_FAILURES):
            
            video = self.video_path_handler.video_from_path(
                os.path.join(self.videos_path, f'{key}.mp4'),
                decode_audio=self._decode_audio,
                decoder=self._decoder,
            )

            loaded_clip = video.get_clip(start, end)

            video_is_null = (loaded_clip is None or loaded_clip["video"] is None)
            
            if video_is_null:
                raise Exception("Failed to load clip {}; trial {}".format(video.name, i_try))

            frames = loaded_clip["video"]
            if self.transform is not None:
                frames = self.transform(frames)

            return frames
        else:
            raise RuntimeError(
                f"Failed to load video {key} after {self._MAX_CONSECUTIVE_FAILURES} retries."
            )
