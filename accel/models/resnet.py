import torch
from torch import nn
import torch.nn.functional as F

from tsai.imports import Module
from tsai.models.layers import ConvBlock, BN1d
from tsai.models.utils import Squeeze, Add
#from .utils import ConvBlock, Conv1dSamePadding

class ResBlock(Module):
    def __init__(self, ni, nf, kss=[7, 5, 3]):
        self.convblock1 = ConvBlock(ni, nf, kss[0])
        self.convblock2 = ConvBlock(nf, nf, kss[1])
        self.convblock3 = ConvBlock(nf, nf, kss[2], act=None)

        # expand channels for the sum if necessary
        self.shortcut = BN1d(ni) if ni == nf else ConvBlock(ni, nf, 1, act=None)
        self.add = Add()
        self.act = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.add(x, self.shortcut(res))
        x = self.act(x)
        return x

class ResNetBody(Module):
    def __init__(self, c_in):
        nf = 64
        kss=[7, 5, 3]
        self.resblock1 = ResBlock(c_in, nf, kss=kss)
        self.resblock2 = ResBlock(nf, nf * 2, kss=kss)
        self.resblock3 = ResBlock(nf * 2, nf * 2, kss=kss)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.squeeze = Squeeze(-1)

    def forward(self, x):
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.squeeze(self.gap(x))
        return x

class ResNet(Module):
    def __init__(self, c_in, c_out):
        nf = 64
        self.body = ResNetBody(c_in)
        self.fc = nn.Linear(nf * 2, c_out)

    def forward(self, x):
        x = self.body(x)
        return self.fc(x)

class SegmentationHead(Module):
    def __init__(self, ni, nf, c_out, kss=[3, 3, 3]):
        self.convblock1 = ConvBlock(ni, nf, kss[0])
        self.convblock2 = ConvBlock(nf, nf, kss[1])
        self.convblock3 = ConvBlock(nf, c_out, kss[2], act=None)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)

        return x

class SegmentationResnet(Module):
    def __init__(self, c_in, c_out):
        nf = 64
        kss=[7, 5, 3]
        self.resblock1 = ResBlock(c_in, nf, kss=kss)
        self.resblock2 = ResBlock(nf, nf * 2, kss=kss)
        self.resblock3 = ResBlock(nf * 2, nf * 2, kss=kss)

        self.seg_head = SegmentationHead(nf*2, nf, c_out)

        # self.gap = nn.AdaptiveAvgPool1d(1)
        # self.squeeze = Squeeze(-1)
        # self.fc = nn.Linear(nf * 2, c_out)

    def forward(self, x):
        x = self.resblock1(x)
        x = self.resblock2(x)
        x1 = self.resblock3(x)
        # x2 = self.squeeze(self.gap(x1))
        return self.seg_head(x1)