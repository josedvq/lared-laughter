import pickle

import torch
import pandas as pd

class FatherDataset():
    def __init__(
        self,
        examples: pd.DataFrame,
        map_datasets:dict,
        id_column = 'id',
        label_column = 'label'
    ) -> None:
        self.examples = examples
        self.map_datasets = map_datasets
        self.id_column = id_column
        self.label_column = label_column

    def __getitem__(self, idx) -> dict:
        key = self.examples[self.id_column][idx]

        item = {}
        for ds_name, ds in self.map_datasets.items():
            item[ds_name] = self.map_datasets[ds_name][key]
        item['label'] = self.examples[self.label_column][idx]
        return item

    def __len__(self):
        return len(self.examples)

    def get_all_labels(self):
        return self.examples[self.label_column].to_numpy().squeeze()

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