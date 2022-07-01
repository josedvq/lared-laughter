from functools import partial
from typing import Tuple
import torch, numpy as np, librosa
from torch.utils import data
from joblib import Parallel, delayed
import pandas as pd
import joblib

import pickle
import torch

from lared_laughter.audio import audio_utils

def _extract_features( audio, sr, feature_fn, min_samples=None, max_samples=None, transform=None):
        # if the file is too short, pad it with zeros
        if min_samples is not None and len(audio) < min_samples:
            audio = np.pad(
                audio,
                pad_width=(0, min_samples-len(audio)))

        if max_samples is not None and len(audio) > max_samples:
            audio = audio[:max_samples]

        features = feature_fn(y=audio, sr=sr)[None,:,:]

        if transform is not None:
            return transform(features)
        return features

class AudioLaughterExtractor():

    def __init__(self, audios_path, sr=8000, min_len=None, max_len=None,
        feature_fn=partial(audio_utils.featurize_melspec, hop_length=186),
        transform=None, n_jobs=1):
        
        self.audios = pickle.load(open(audios_path, 'rb'))
        self.sr = sr

        if min_len is not None and max_len is not None:
            assert max_len >= min_len
        self.min_samples = round(sr*min_len) if min_len is not None else None # convert to samples
        self.max_samples = round(sr*max_len) if max_len is not None else None

        self.feature_fn = feature_fn
        self.transform = transform
        self.n_jobs = n_jobs

    def _subsample_audio(self, audio, window: Tuple[int, int]):
        
        start = max(0, round(window[0] * self.sr))
        end = min(len(audio), round(window[1] * self.sr))

        audio = audio[start: end]

        return audio

    def __call__(self, key, start=None, end=None):
        assert (start is None and end is None) or (start is not None and end is not None)
        audio = self.audios[key]

        if start is not None and end is not None:
            audio = self._subsample_audio(audio, (start, end))

        # if the file is too short, pad it with zeros
        if self.min_samples is not None and len(audio) < self.min_samples:
            audio = np.pad(
                audio,
                pad_width=(0, self.min_samples-len(audio)))

        if self.max_samples is not None and len(audio) > self.max_samples:
            audio = audio[:self.max_samples]

        features = self.feature_fn(y=audio, sr=self.sr)[None,:,:]

        if self.transform is not None:
            return self.transform(features)
        return features

    def extract_multiple(self, keys):
        audios = [self.audios[k[0]] for k in keys]

        for i, (_, start, end) in enumerate(keys):
            assert (start is None and end is None) or (start is not None and end is not None)

            if start is not None and end is not None:
                audios[i] = self._subsample_audio(audios[i], (start, end))

        # parallel feature extraction
        # audio, sr, feature_fn, min_samples=None, max_samples=None, transform=None
        features = Parallel(n_jobs=self.n_jobs)(
            delayed(_extract_features)(
                audio,
                sr=self.sr,
                feature_fn = self.feature_fn,
                min_samples = self.min_samples,
                max_samples = self.max_samples,
                transform = self.transform
            ) for audio in audios)
        return np.stack(features)
        
        # return np.stack([self(*k) for k in keys])

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
