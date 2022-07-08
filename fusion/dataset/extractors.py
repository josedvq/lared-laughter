import os
import pickle
from typing import Tuple

import torch
import numpy as np

class CacheExtractor():
    def __init__(self, model, extractor, cache_path, enable=True):
        self.model = model
        if model is not None:
            self.device = next(model.parameters()).get_device()
        else:
            self.device = None
        self.extractor = extractor
        self.cache_path = cache_path
        if os.path.exists(cache_path):
            assert os.path.isdir(cache_path)
        else:
            os.makedirs(cache_path)
        self.enabled = enable
        self.clear()

    def clear(self):
        self.cache = None
        self.cache_name = None
        self.dirty = False

    def _get_fpath(self, fname):
        if type(fname) == int:
            fname = f'fold{fname}.pkl'
        elif type(fname) == str:
            pass
        else:
            raise ValueError('Unrecognized fname type')

        return os.path.join(self.cache_path, fname)

    def load(self, fname='cache.pkl'):
        fpath = self._get_fpath(fname)
        if self.enabled:
            if os.path.exists(fpath):
                self.cache = pickle.load(open(fpath, 'rb'))
            else:
                self.cache = {}
            self.cache_name = fname

    def store(self, clear=False):
        if self.enabled and self.dirty:
            assert self.cache_name is not None
            fpath = self._get_fpath(self.cache_name)
            pickle.dump(self.cache, open(fpath, 'wb'))

        if clear:
           self.clear()

    def extract_multiple(self, keys):
        if self.enabled and all([k in self.cache for k in keys]):
            return np.stack([self.cache[k] for k in keys])
        else:
            if self.model is None:
                raise Exception('key not in cache and model not set.')

            X = torch.tensor(self.extractor.extract_multiple(keys)).to(device=self.device)
            assert len(X) == len(keys)

            X = self.model(X).cpu()

            # cache the data
            if self.enabled:
                for i, k in enumerate(keys):
                    self.cache[k] = X[i].numpy()
                self.dirty = True
            
            return X


class LabelExtractor():
    def __init__(self, annot_path, min_len=None, transform=None):
        self.annot_path = annot_path
        self.transform = transform
        self.min_len = min_len
        
        self.annot = pickle.load(open(annot_path, 'rb'))

    def __call__(self, key):
        example_annot = self.annot[key]

        if self.min_len is not None and example_annot.shape[0] < self.min_len:
            example_annot = np.pad(example_annot, 
                (0, self.min_len-example_annot.shape[0]),
                mode='constant',
                constant_values= 0)

        if self.transform:
            example_annot = self.transform(example_annot)

        return example_annot


class SegMaskExtractor():
    def __init__(self, annot_path, transform=None, min_len=None, max_len=None, sr=30):
        self.annot_path = annot_path
        self.transform = transform
        if min_len is not None and max_len is not None:
            assert max_len >= min_len
        self.min_samples = round(sr*min_len) if min_len is not None else None
        self.max_samples = round(sr*max_len) if max_len is not None else None
        
        self.sr = sr
        
        self.annot = pickle.load(open(annot_path, 'rb'))

    def _subsample(self, annot, window: Tuple[int, int]):
        start = max(0, round(window[0] * self.sr))
        end  = min(len(annot), round(window[1] * self.sr))

        if len(annot) < end - start:
            raise Exception(f'Not implemented. annot shape is {str(annot.shape)}')

        annot = annot[start: end]
        return annot

    def extract_multiple(self, keys):
        return np.stack([self(*k) for k in keys])

    def __call__(self, key, start=None, end=None):
        assert (start is None and end is None) or (start is not None and end is not None)

        example_annot = self.annot[key].astype(np.float32)

        if start is not None and end is not None:
            example_annot = self._subsample(example_annot, (start, end))

        if self.min_samples is not None and len(example_annot) < self.min_samples:
            example_annot = np.pad(example_annot, 
                ((0, self.min_samples - len(example_annot))),
                mode='constant',
                constant_values= 0)

        if self.max_samples is not None and len(example_annot) > self.max_samples:
            example_annot = example_annot[:self.max_samples]

        if self.transform:
            return self.transform(example_annot)

        return example_annot