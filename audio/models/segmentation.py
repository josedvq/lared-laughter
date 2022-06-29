import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .resnet import ResidualBlock, ResNetBody

class AudioSegmentationResnet(nn.Module):
    def __init__(self,
        num_classes=1,
        dropout_rate=0.5,
        filter_sizes=[64,32,16,16],
        output_size=60):
        super(AudioSegmentationResnet, self).__init__()

        self.body = ResNetBody(filter_sizes=filter_sizes)

        self.head = nn.Sequential(
            ResidualBlock(filter_sizes[3], 8, stride=(1,2)),
            ResidualBlock(8, num_classes, stride=(1,2)),
            nn.AvgPool2d((1,4), stride=1),
        )

        # at this point the output should have size=1 along the mel dimension
        self.upsample = nn.Sequential(
            nn.Flatten(start_dim=2),
            nn.Upsample(size=(output_size), mode='linear')
        )
    
    def forward(self, x):
    # Output of one layer becomes input to the next
        x = self.body(x)
        x = self.head(x)
        # at this point the output should have size=1 along the mel dimension
        assert x.shape[-1] == 1
        x = self.upsample(x)
        return x.squeeze()

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