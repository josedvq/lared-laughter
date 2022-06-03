from typing import Tuple
from math import ceil
import pickle
import torch
import numpy as np

from scipy.interpolate import interp1d

class AccelExtractor():
    def __init__(self, accel_path, transform=None, min_len=None):
        self.accel_path = accel_path
        self.transform = transform
        self.min_len = min_len
        self.num_channels = 3
        self.blank_len = 20*15
        
        self.accel = pickle.load(open(accel_path, 'rb'))

    def _subsample(self, accel, window: Tuple[int, int]):
        start = round(window[0] * 20)
        end  = round(window[1] * 20)

        if accel.shape[1] < end - start:
            raise Exception(f'Not implemented. Accel shape is {str(accel.shape)}')

        accel = accel[:, start: end]
        return accel

    def __call__(self, key, start=None, end=None):
        try:
            example_accel = self.accel[key].transpose().astype(np.float32)
        except KeyError as err:
            if self.min_len is not None:
                # return a large blank example when there is no accel.
                example_accel = np.zeros((self.num_channels, self.blank_len), dtype=np.float32)
            else:
                raise err

        if self.min_len is not None and example_accel.shape[1] < self.min_len:
            example_accel = np.pad(example_accel, 
                ((0, 0), (0, self.min_len-example_accel.shape[0])),
                mode='constant',
                constant_values= 0)

        if start is not None and end is not None:
            example_accel = self._subsample(example_accel, (start, end))

        if self.transform:
            return self.transform(example_accel)

        return example_accel
            
class AccelLaughterDataset(torch.utils.data.Dataset):
    """Loads a dataset of laughter examples"""

    def __init__(self, examples_df, accel_path, annot_path, example_len=64, label='pressed_key', pad=True, transform=None):
        """
        Args:
            examples_df (string): dataframe with the example info
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.examples_df = examples_df
        self.accel_path = accel_path
        self.transform = transform
        self.example_len = example_len
        self.label = label
        self.pad = pad
        
        # load accel data
        self.accel = pickle.load(open(accel_path, 'rb'))
        print(f'loaded {len(self.examples_df)} examples')

        # filter out examples with no accel
        self.examples_df = self.examples_df[self.examples_df.apply(lambda x: x['hash'] in self.accel, axis=1)]
        print(f'{len(self.examples_df)} have accel')

        # load continuous annotations
        self.annot = pickle.load(open(annot_path, 'rb'))

        # filter out examples with no annot
        self.examples_df = self.examples_df[self.examples_df.apply(
            lambda x: (x['hit_id'], x['hash'], x['condition']) in self.annot, axis=1)]
        print(f'{len(self.examples_df)} have annotations')

        self.examples_df.reset_index(inplace=True)

        
    def __len__(self):
        return len(self.examples_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        example = self.examples_df.iloc[idx,:]
        example_accel = self.accel[example['hash']]

        annot = self.annot[(example['hit_id'], example['hash'], example['condition'])]

        res = {'accel': example_accel, 'seg_mask': annot}
        if self.transform:
            return self.transform(res)

        return res

    def get_all_labels(self):
        return self.examples_df[self.label].to_numpy().squeeze()

