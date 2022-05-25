import torch, numpy as np, librosa
from torch.utils import data
import audio_utils
from joblib import Parallel, delayed
import pandas as pd

import pickle
import torch
            
class SwitchBoardLaughterDataset(torch.utils.data.Dataset):
    def __init__(self, df, audios, feature_fn, sr, subsample_length=-1, id_column='id', label_column='label'):
        # For training, we should set subsample to True, for val/testing, set to false
        # When subsample is False, we use the data in 'subsampled_offset/duration' every time
        # When it's True, we re-subsample to get more variation in time

        self.audios = audios
        self.subsample_length=subsample_length

        self.notfound = [id for id in df[id_column] if id not in self.audios]
        print(f'df: {len(df)}, audios: {len(audios)}, not found: {len(self.notfound)}')

        self.df = df[~df[id_column].isin(self.notfound)].reset_index()
        print(f'df: {len(self.df)}, audios: {len(audios)}, not found: {len(self.notfound)}')

        # Columns: [region start, region duration, subsampled region start, subsampled region duration, audio path, label]
        #self.df = pd.read_csv(data_file,sep='\t',header=None,
        #    names=['offset','duration','subsampled_offset','subsampled_duration','audio_path','label'])
        self.feature_fn = feature_fn
        self.sr = sr
        self.id_column = id_column
        self.label_column = label_column

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        id = self.df[self.id_column][index]
        audio_file = self.audios[id]
        
        start = 0
        end = self.df.end_time[index]
        duration = end - start

        if self.subsample_length != -1:

            # if the file is too short, pad it with zeros
            min_samples = round(self.sr * self.subsample_length)
            if len(audio_file) < min_samples:
                audio_file = np.pad(
                    audio_file,
                    pad_width=(0, min_samples-len(audio_file)))

            audio_file_length = librosa.core.samples_to_time(len(audio_file), sr=self.sr)
            start, duration = audio_utils.subsample_time(start, duration, audio_file_length=audio_file_length,
                subsample_length=1.0, padding_length=0.5)

        
        X = self.feature_fn(y=audio_file, sr=self.sr, offset=start, duration=duration)[None,:,:]
        y = int(self.df[self.label_column][index])
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
