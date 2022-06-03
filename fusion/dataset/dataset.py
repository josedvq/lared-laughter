import pickle

import torch
import pandas as pd

class FatherDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        examples: pd.DataFrame,
        extractors:dict,
        id_column = 'id',
        transform = None
    ) -> None:
        self.examples = examples
        self.extractors = extractors
        self.id_column = id_column
        self.transform = transform

    def __getitem__(self, idx) -> dict:
        item = {}
        for ex_name, extractor in self.extractors.items():
            id_column = extractor.get('id_column', self.id_column)
            if callable(id_column):
                key = id_column(self.examples.iloc[idx, :])
            else:
                key = self.examples[id_column][idx]

            item[ex_name] = extractor['extractor'][key]

        if self.transform is not None:
            return self.transform(item)

        return item

    def __len__(self):
        return len(self.examples)

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