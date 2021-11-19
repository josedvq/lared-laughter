import pickle
import torch
            
class AccelLaughterDataset(torch.utils.data.Dataset):
    """Loads a dataset of laughter examples"""

    def __init__(self, examples_df, accel_path, transform=None):
        """
        Args:
            examples_df (string): dataframe with the example info
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.examples_df = examples_df
        self.accel_path = accel_path
        self.transform = transform
        
        self.accel = pickle.load(open(accel_path, 'rb'))
        print(f'loaded {len(self.examples_df)} examples')

        self.examples_df = self.examples_df[self.examples_df.apply(lambda x: x['hash'] in self.accel, axis=1)]
        print(f'{len(self.examples_df)} have accel')

        self.examples_df.reset_index(inplace=True)

        
    def __len__(self):
        return len(self.examples_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        example = self.examples_df.iloc[idx,:]
        print(example)
        example_accel = self.accel[example['hash']]

        if self.transform:
            example_accel = self.transform(example_accel)

        return example_accel, example['pressed_key']
