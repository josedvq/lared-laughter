import os
import pickle

import torch
import numpy as np

class CacheExtractor():
    def __init__(self, model, extractor, cache_path, enable=True):
        self.model = model
        self.device = next(model.parameters()).get_device()
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
        if self.enabled:
            assert self.cache_name is not None
            fpath = self._get_fpath(self.cache_name)
            pickle.dump(self.cache, open(fpath, 'wb'))

        if clear:
           self.clear()

    def __call__(self, key, start, end) -> dict:
        if (key, start, end) in self.cache:
            return self.cache[(key, start, end)]
        else:
            pass

    def extract_multiple(self, keys):
        if self.enabled and all([k in self.cache for k in keys]):
            return np.stack([self.cache[k] for k in keys])
        else:
            X = torch.tensor(self.extractor.extract_multiple(keys)).to(device=self.device)
            assert len(X) == len(keys)

            X = self.model(X).cpu()

            # cache the data
            if self.enabled:
                for i, k in enumerate(keys):
                    self.cache[k] = X[i].numpy()
            
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