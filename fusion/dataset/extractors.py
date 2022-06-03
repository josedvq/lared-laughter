import pickle
import numpy as np

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