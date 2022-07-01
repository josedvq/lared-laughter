from typing import Tuple
from math import ceil
import pickle
import torch
import numpy as np

from scipy.interpolate import interp1d

class AccelExtractor():
    def __init__(self, accel_path, transform=None, min_len=None, max_len=None, sr=20):
        self.accel_path = accel_path
        self.transform = transform
        if min_len is not None and max_len is not None:
            assert max_len >= min_len
        self.min_samples = round(sr*min_len) if min_len is not None else None
        self.max_samples = round(sr*max_len) if max_len is not None else None
        
        self.num_channels = 3
        self.blank_len = 20*15
        
        self.accel = pickle.load(open(accel_path, 'rb'))

    def _subsample(self, accel, window: Tuple[int, int]):
        start = max(0, round(window[0] * 20))
        end  = min(accel.shape[1], round(window[1] * 20))

        if accel.shape[1] < end - start:
            raise Exception(f'Not implemented. Accel shape is {str(accel.shape)}')

        accel = accel[:, start: end]
        return accel

    def extract_multiple(self, keys):
        return np.stack([self(*k) for k in keys])

    def __call__(self, hash, start=None, end=None):
        assert (start is None and end is None) or (start is not None and end is not None)

        try:
            example_accel = self.accel[hash].transpose().astype(np.float32)
        except KeyError as err:
            if self.min_samples is not None:
                # return a large blank example when there is no accel.
                example_accel = np.zeros((self.num_channels, self.blank_len), dtype=np.float32)
            else:
                raise err

        if start is not None and end is not None:
            example_accel = self._subsample(example_accel, (start, end))

        if self.min_samples is not None and example_accel.shape[1] < self.min_samples:
            example_accel = np.pad(example_accel, 
                ((0, 0), (0, self.min_samples-example_accel.shape[1])),
                mode='constant',
                constant_values= 0)

        if self.max_samples is not None and example_accel.shape[1] > self.max_samples:
            example_accel = example_accel[:, :self.max_samples]

        if self.transform:
            return self.transform(example_accel)

        return example_accel
