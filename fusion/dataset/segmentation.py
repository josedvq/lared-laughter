
import torch
import pandas as pd
import numpy as np
from .extractors import LabelExtractor
from scipy.interpolate import interp1d

class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        examples: pd.DataFrame,
        annot_path: str,
        extractors:dict,
        window_len: int = None,
        resize_len: int = None,
        id_column = 'id',
        transform = None
    ) -> None:
        self.examples = examples
        self.labels = LabelExtractor(annot_path, min_len=90)
        self.extractors = extractors
        self.window_len = window_len
        self.resize_len = resize_len

        self.id_column = id_column
        self.transform = transform

    def _get_label_key(self, x):
        return (x['hit_id'], x['hash'], x['condition'])

    def _subsample(self, seg_mask):
        if np.count_nonzero(seg_mask) > 0:
            # take candidates for window center around positive chunks
            window_candidates = np.nonzero(seg_mask)[0]
        else:
            window_candidates = np.arange(0, len(seg_mask))

        # remove candidates too close to the ends.
        half_window = self.window_len // 2
        window_candidates = window_candidates[window_candidates > half_window]
        window_candidates = window_candidates[window_candidates < len(seg_mask) - half_window]
        if len(window_candidates) == 0:
            window_candidates = np.arange(half_window, len(seg_mask)-half_window+1)

        center = np.random.choice(window_candidates)
        start = center - half_window
        end = start+self.window_len
        
        return start, end

    def _interp_mask(self, seg_mask, first, last):
        f = interp1d(range(len(seg_mask)), seg_mask, kind='nearest', fill_value='extrapolate')

        seg_mask = f(np.linspace(first, last, num=self.resize_len))
        return seg_mask

    def __getitem__(self, idx) -> dict:

        ex = self.examples.iloc[idx,:]
        seg_mask = self.labels(self._get_label_key(ex))

        if self.window_len is not None:
            start, end = self._subsample(seg_mask)
            start_time, end_time = start / 30,  end / 30

            if end-start != self.resize_len:
                seg_mask = self._interp_mask(seg_mask, start, end)
            else:
                seg_mask = seg_mask[start: end]
        else:
            start_time, end_time = None, None

        item = {}
        for ex_name, extractor in self.extractors.items():
            id_column = extractor.get('id_column', self.id_column)
            if callable(id_column):
                key = id_column(self.examples.iloc[idx, :])
            else:
                key = self.examples[id_column][idx]

            item[ex_name] = extractor['extractor'](key, start_time, end_time)

        
        item['seg_mask'] = seg_mask

        if self.transform is not None:
            return self.transform(item)

        return item

    def __len__(self):
        return len(self.examples)