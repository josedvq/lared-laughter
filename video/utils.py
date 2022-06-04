import os
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import pytorchvideo
import pytorch_lightning

from transforms import get_kinetics_train_transform, get_kinetics_val_transform



def get_metrics(outputs, labels, type='binary'):
    if type == 'binary':
        proba = torch.sigmoid(outputs)
        pred = (proba > 0.5)

        correct = pred.eq(outputs.bool()).sum().item()
        return {
            'auc': roc_auc_score(labels, proba),
            'correct': correct
        }
    elif type == 'regression':
        return {
            'mse': torch.nn.functional.mse_loss(outputs, labels, reduction='mean'),
            'l1': torch.nn.functional.l1_loss(outputs, labels, reduction='mean')
        }

class KineticsDataModule(pytorch_lightning.LightningDataModule):

  # Dataset configuration
  _CLIP_DURATION = 2  # Duration of sampled clip for each video
  _BATCH_SIZE = 8
  _NUM_WORKERS = 4  # Number of parallel processes fetching data

  def __init__(self, data_path, spatial_res=244, pack=False):
    super().__init__()
    self.data_path = data_path
    self.pack = pack
    self.subsample_frames = 64 if pack else 8
    self.spatial_res = spatial_res


  def train_dataloader(self):
    """
    Create the Kinetics train partition from the list of video labels
    in {self._DATA_PATH}/train
    """

    train_dataset = my_video_dataset(
        data_path=os.path.join(self.data_path, "train.csv"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("random", self._CLIP_DURATION),
        transform=get_kinetics_train_transform(),
        decode_audio=False
    )
    return torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        batch_size=self._BATCH_SIZE,
        num_workers=self._NUM_WORKERS,
    )

  def val_dataloader(self):
    """
    Create the Kinetics validation partition from the list of video labels
    in {self._DATA_PATH}/val
    """
   
    val_dataset = pytorchvideo.data.Kinetics(
        data_path=os.path.join(self._DATA_PATH, "val.csv"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", self._CLIP_DURATION),
        transform=get_kinetics_val_transform(),
        decode_audio=False
    )
    return torch.utils.data.DataLoader(
        val_dataset,
        batch_size=self._BATCH_SIZE,
        num_workers=self._NUM_WORKERS,
    )