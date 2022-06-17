import os
import yaml
import pathlib
import pickle

import torch
import torch.nn as nn
import pytorchvideo.models

from lared_laughter.constants import models_path
from slowfast.models.ptv_model_builder import PTVResNet, PTVSlowFast
from slowfast.config.defaults import get_cfg

# MODELS
# slowfast - pytorchvideo models are here:
# https://github.com/facebookresearch/SlowFast/blob/main/projects/pytorchvideo/README.md

def _get_resnet_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(
        pathlib.Path(__file__).parent.resolve(), 'configs', 
        'SLOW_8x8_R50.yaml'))
    return cfg

class SegmentationHead(nn.Module):
    def __init__(self, output_size):
        super(SegmentationHead, self).__init__()

        self.head = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 8, 8), stride=(1, 1, 1), padding=(0, 0, 0)),
            nn.Conv3d(2048, 1, (3,1,1), stride=1, padding=(1,0,0))
        )

        self.upsample = nn.Sequential(
            nn.Flatten(start_dim=2),
            nn.Upsample(size=(output_size), mode='linear')
        )

    def forward(self, x):
        x = self.head(x)
        assert x.shape[-1] == 1
        x = self.upsample(x)
        return x.squeeze()


def make_slow_pretrained_segmenter(output_size=60):
    cfg = _get_resnet_cfg()
    model = PTVResNet(cfg, head=SegmentationHead(output_size))

    chckpt = torch.load(os.path.join(models_path, 'SLOW_8x8_R50.pyth'))

    model.load_state_dict(chckpt['model_state'], strict=False)
    for param in model.parameters():
        param.requires_grad = False

    for param in model.head.parameters():
        param.requires_grad = True

    return model

def make_resnet_pretrained_classifier():

    cfg = _get_resnet_cfg()
    model = PTVResNet(cfg)

    chckpt = torch.load(os.path.join(models_path, 'SLOW_8x8_R50.pyth'))

    model.load_state_dict(chckpt['model_state'])
    for param in model.parameters():
        param.requires_grad = False

    model.model.blocks[-1].proj = nn.Linear(2048, 2)

    return model

def make_slowfast_pretrained_classifier():

    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(
        pathlib.Path(__file__).parent.resolve(), 'configs', 
        'SLOWFAST_8x8_R50.yaml'))

    model = PTVSlowFast(cfg)

    chckpt = torch.load(os.path.join(models_path, 'SLOWFAST_8x8_R50.pyth'))

    model.load_state_dict(chckpt['model_state'])
    for param in model.parameters():
        param.requires_grad = False

    model.model.blocks[-1].proj = nn.Linear(2304, 2)

    return model

def make_slowfast_feature_extractor():
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.blocks[-1].proj = nn.Linear(2304, 2)

    return model

def make_resnet_feature_extractor():
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.blocks[-1].proj = nn.Linear(2048, 2)

    return model