import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .resnet import ResNetBody

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=1,dropout_rate=0.5,linear_layer_size=192,filter_sizes=[64,32,16,16]):
        super().__init__()
        print(f"training with dropout={dropout_rate}")
        
        self.body = ResNetBody(filter_sizes=filter_sizes)

        self.bn2 = nn.BatchNorm1d(linear_layer_size)
        self.bn3 = nn.BatchNorm1d(32)
        self.linear1 = nn.Linear(linear_layer_size, 32)
        self.linear2 = nn.Linear(32, num_classes)
      
        self.dropout = nn.Dropout(dropout_rate)
        
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = np.inf

    def forward(self, x):
    # Output of one layer becomes input to the next
        x = self.body(x)
        x = nn.AvgPool2d(4)(x)
        x = x.view(x.size(0), -1)
        x = self.bn2(x)
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.bn3(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

def get_pretrained_classifier(num_classes=1):
    model = ResNetClassifier(
        num_classes=num_classes)
        
    dir_path = os.path.dirname(os.path.realpath(__file__))
    chkpt = torch.load(os.path.join(dir_path, 
        './pretrained_audioset/audioset-epoch=09-val_loss=0.46.ckpt'))

    sd = OrderedDict()
    for el in chkpt['state_dict']:
        sd[el[6:]] = chkpt['state_dict'][el]

    res = model.body.load_state_dict(sd, strict=False)
    print(f'loaded pre-trained model')
    print(f'missing keys {str(res.missing_keys)}')
    print(f'unexpected keys {str(res.unexpected_keys)}')

    for param in model.parameters():
        param.requires_grad = False
        
    # train the head only
    for param in model.head.parameters():
        param.requires_grad = True

    return model