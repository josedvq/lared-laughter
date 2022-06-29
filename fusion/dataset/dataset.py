import pickle
import random

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .extractors import LabelExtractor

class FatherDatasetSubset(torch.utils.data.Subset):
    def auc(self, idxs, proba) -> float:
        return self.dataset.auc(idxs, proba)

    def accuracy(self, idxs, proba) -> float:
        return self.dataset.accuracy(idxs, proba)

class FatherDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        examples: pd.DataFrame,
        extractors:dict,
        label_column: str,
        id_column = 'id',
        transform = None,
        eval=True
    ) -> None:
        self.examples = examples
        self.extractors = extractors
        self.label_column = label_column
        self.id_column = id_column
        self.transform = transform
        self.eval = eval

    def _get_start_end(self, ex):
        onsets = ex['onset_times'] if len(ex['onset_times']) > 0 else None
        offsets= ex['offset_times'] if len(ex['offset_times']) > 0 else None

        example_len = ex['_end_time'] - ex['_ini_time']

        if self.eval:
            # dataset in evaluation mode
            if onsets is not None:
                # center the window around the first detection
                start, end = onsets[0], offsets[0]
                mid = (end+start) / 2
            else:
                # for negative examples, select the middle section
                mid = example_len / 2
        else:
            # dataset in train mode
            if onsets is not None:
                # select a window centered within a laughter ep
                start, end = random.choice(list(zip(onsets, offsets)))
                mid = np.random.uniform(start, end)
            else:
                # for negative examples, select a section randomly
                mid = np.random.uniform(0, example_len)

        start = max(0, mid-0.75)
        end   = min(example_len, mid+0.75)
        return start, end

    def __getitem__(self, idx) -> dict:
        ex = self.examples.iloc[idx,:]
        start, end = self._get_start_end(ex)

        item = {}
        for ex_name, extractor in self.extractors.items():
            id_column = extractor.get('id_column', self.id_column)
            if callable(id_column):
                key = id_column(self.examples.iloc[idx, :])
            else:
                key = self.examples[id_column][idx]

            item[ex_name] = extractor['extractor'](key, start=start, end=end)

        item['label'] = ex[self.label_column]
        item['intval']= [start, end]
        item['index'] = idx

        if self.transform is not None:
            return self.transform(item)

        return item

    def __len__(self):
        return len(self.examples)

    def get_all_labels(self):
        return self.examples[self.label_column].to_numpy()

    def auc(self, idxs, proba: np.array):
        labels = self.get_all_labels()
        labels = labels[idxs]
        assert len(labels) == len(proba)
        return roc_auc_score(labels, proba)

    def accuracy(self, idxs, proba: np.array):
        labels = self.get_all_labels()
        labels = labels[idxs]
        assert len(labels) == len(proba)
        pred = np.argmax(proba, axis=1)

        correct = (pred == labels).sum().item()
        return correct / len(labels)

class FeatureDataset():
    def __init__(
        self,
        features_path: str,
        key = None
    ) -> None:

        self.features = pickle.load(open(features_path, 'rb'))
        if key is not None:
            self.features = self.features[key]

    def __getitem__(self, key):
        return torch.tensor(self.features[key])