import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from resnet import ResidualBlock, ResidualBlockNoBN

class AudioSegmentationResnet(nn.Module):
    def __init__(self,
        num_classes=1,
        dropout_rate=0.5,
        linear_layer_size=192,
        filter_sizes=[64,32,16,16],
        output_size=90):
        super(AudioSegmentationResnet, self).__init__()

        # Initial input conv
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=(3, 3),
            stride=1, padding=1, bias=False
        )

        self.bn1 = nn.BatchNorm2d(64)
        
        self.linear_layer_size=linear_layer_size
        
        self.filter_sizes = filter_sizes

        self.block1 = self._create_block(64, filter_sizes[0], stride=1)
        self.block2 = self._create_block(filter_sizes[0], filter_sizes[1], stride=2)
        self.block3 = self._create_block(filter_sizes[1], filter_sizes[2], stride=2)
        self.block4 = self._create_block(filter_sizes[2], filter_sizes[3], stride=2)


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
    
    # A block is just two residual blocks for ResNet18
    def _create_block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride),
            ResidualBlock(out_channels, out_channels, 1)
        )

    def forward(self, x):
    # Output of one layer becomes input to the next
        x = nn.ReLU()(self.bn1(self.conv1(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.head(x)
        # at this point the output should have size=1 along the mel dimension
        assert x.shape[-1] == 1
        x = self.upsample(x)
        return x.squeeze()