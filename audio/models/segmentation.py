import os
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .resnet import ResidualBlock, ResNetBody



def get_pretrained_segmentation_model(output_size=60):
    model = AudioSegmentationResnet(
        filter_sizes=[64,32,16,16],
        output_size=output_size)
        
    dir_path = os.path.dirname(os.path.realpath(__file__))
    chkpt = torch.load(os.path.join(dir_path, 
        './pretrained_audioset/audioset-epoch=09-val_loss=0.46.ckpt'))

    sd = OrderedDict()
    for el in chkpt['state_dict']:
        sd[el[6:]] = chkpt['state_dict'][el]

    model.load_state_dict(sd, strict=False)

    for param in model.parameters():
        param.requires_grad = False
        
    # train the head only
    for param in model.head.parameters():
        param.requires_grad = True

    return model