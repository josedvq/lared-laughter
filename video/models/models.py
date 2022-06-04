import os
import yaml

import torch
import torch.nn as nn
import pytorchvideo.models

from lared_laughter.constants import models_path
from slowfast.models.ptv_model_builder import PTVResNet

def make_slow_pretrained_segmenter():
    
    model = pytorchvideo.models.slowfast.create_slowfast(

    )

def make_resnet_pretrained_classifier():

    cfg = yaml.safe_load(open('./configs/SLOWFAST_8x8_R50.yaml', 'r'))

    model = PTVResNet(cfg)

    # model = pytorchvideo.models.resnet.create_resnet()

    chckpt = torch.load(os.path.join(models_path, 'SLOW_4x16_R50.pyth'))

    model.load_state_dict(chckpt['model_state'])
    for param in model.parameters():
        param.requires_grad = False

    # model.blocks[-1].proj = nn.Linear(2048, 2)

    return model

def make_slowfast_pretrained_classifier():
    model = pytorchvideo.models.slowfast.create_slowfast()

    chckpt = torch.load(os.path.join(models_path, 'SLOWFAST_8x8_R50.pyth'))

    model.load_state_dict(chckpt['model_state'])
    for param in model.parameters():
        param.requires_grad = False

    model.blocks[-1].proj = nn.Linear(2304, 2)

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