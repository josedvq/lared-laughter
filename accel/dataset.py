import pickle
import torch
            
class AccelLaughterDataset(torch.utils.data.Dataset):
    """Loads a dataset of laughter examples"""

    def __init__(self, examples_df, accel_path, example_len=64, label='pressed_key', pad=True, transform=None):
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
        
        self.accel = pickle.load(open(accel_path, 'rb'))
        print(f'loaded {len(self.examples_df)} examples')

        # filter out examples with no accel
        self.examples_df = self.examples_df[self.examples_df.apply(lambda x: x['hash'] in self.accel, axis=1)]
        print(f'{len(self.examples_df)} have accel')

        self.examples_df.reset_index(inplace=True)

        
    def __len__(self):
        return len(self.examples_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        example = self.examples_df.iloc[idx,:]
        example_accel = torch.tensor(self.accel[example['hash']])

        # cut if the example is larger than example_len
        if example_accel.shape[0] >= self.example_len:
            example_accel = example_accel[:self.example_len, :]
        else:
            # resize or pad
            if self.pad:
                pad = self.example_len - example_accel.shape[0]
                example_accel = torch.nn.functional.pad(example_accel, pad=[0,0,0,pad],
                                                    mode='constant', value=0)
            else:
                raise Exception('Not implemented')

        if self.transform:
            example_accel = self.transform(example_accel)

        return example_accel, example[self.label]

    def get_all_labels(self):
        return self.examples_df[self.label].to_numpy().squeeze()

