import torch, numpy as np, librosa
from torch.utils import data
import audio_utils
from joblib import Parallel, delayed
import pandas as pd

class SwitchBoardLaughterDataset(torch.utils.data.Dataset):
    def __init__(self, df, audios, feature_fn, sr, subsample=True):
        # For training, we should set subsample to True, for val/testing, set to false
        # When subsample is False, we use the data in 'subsampled_offset/duration' every time
        # When it's True, we re-subsample to get more variation in time

        self.df = df
        self.audios = audios
        self.subsample=subsample

        self.notfound = [id for id in self.df.yt_id if id not in self.audios]
        print(f'df: {len(df)}, audios: {len(audios)}, not found: {len(self.notfound)}')

        # Columns: [region start, region duration, subsampled region start, subsampled region duration, audio path, label]
        #self.df = pd.read_csv(data_file,sep='\t',header=None,
        #    names=['offset','duration','subsampled_offset','subsampled_duration','audio_path','label'])
        self.feature_fn = feature_fn
        self.sr = sr

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        yt_id = self.df.yt_id[index]
        audio_file = self.audios[yt_id]
        
        start = 0
        end = self.df.end_time[index]
        duration = end - start

        if self.subsample:
            audio_file_length = librosa.core.samples_to_time(len(audio_file), sr=self.sr)
            start, duration = audio_utils.subsample_time(start, duration, audio_file_length=audio_file_length,
                subsample_length=1.0, padding_length=0.5)

        
        X = self.feature_fn(y=audio_file, sr=self.sr, offset=start, duration=duration)
        y = self.df.laughter[index]
        return X, y

class SwitchBoardLaughterInferenceDataset(torch.utils.data.Dataset):
    def __init__(self, audio_path, feature_fn, sr=8000, n_frames=44):
        self.audio_path = audio_path
        self.n_frames = n_frames
        self.feature_fn = feature_fn
        self.sr = sr
        self.n_frames = n_frames

        self.y, _ = librosa.load(audio_path, sr=sr)
        self.features = feature_fn(y=self.y,sr=self.sr)

    def __len__(self):
        return len(self.features)-self.n_frames

    def __getitem__(self, index):
        # return None for labels
        return (self.features[index:index+self.n_frames], None)
