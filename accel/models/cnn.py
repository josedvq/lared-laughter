from collections import OrderedDict

import torch
import torch.nn as nn

# Here we define our model as a class
class MyAlexNet(nn.Module):
    """
    Input - 3x60
    C1 - 24@56 (5 kernel, stride=1)
    S1 - 24@27 (3 kernel, stride 2) Subsampling

    C2 - 24@27 (3 kernel, stride=1, padding=1)
    S2 - 64@13 (3 kernel, stride 2) Subsampling

    C3 - 96@13 (3 kernel, stride=1, padding=1)
    C4 - 96@13 (3 kernel, stride=1, padding=1)
    C5 - 64@13 (3 kernel, stride=1, padding=1)
    S3 - 64@6 (3 kernel, stride 2) Subsampling

    F1 - 448 > 224
    F2 - 224 > 224
    F3 - 224 > 1

    F1 - 384 > 384 (64*6)
    F2 - 192 > 192
    F3 - 192 > 1
    """
    def __init__(self):
        super(MyAlexNet, self).__init__()

        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv1d(3, 24, kernel_size=5)),
            ('relu1', nn.ReLU()),
            ('s1', nn.MaxPool1d(kernel_size=3, stride=2)),

            ('c2', nn.Conv1d(24, 64, kernel_size=3, padding=1)),
            ('relu2', nn.ReLU()),
            ('s2', nn.MaxPool1d(kernel_size=3, stride=2)),

            ('c3', nn.Conv1d(64, 96, kernel_size=3, padding=1)),
            ('relu3', nn.ReLU()),

            ('c4', nn.Conv1d(96, 96, kernel_size=3, padding=1)),
            ('relu4', nn.ReLU()),

            ('c5', nn.Conv1d(96, 64, kernel_size=3, padding=1)),
            ('relu5', nn.ReLU()),
            ('s3', nn.MaxPool1d(kernel_size=3, stride=2))
        ]))

        #self.avgpool = nn.AdaptiveAvgPool1d(7)

        self.fc = nn.Sequential(OrderedDict([
            ('d1', nn.Dropout()),
            ('f1', nn.Linear(384, 192)),
            ('relu6', nn.ReLU()),

            ('d2', nn.Dropout()),
            ('f2', nn.Linear(192, 192)),
            ('relu7', nn.ReLU()),

            ('f3', nn.Linear(192, 1))
        ]))

    def forward(self, x):
        x = self.convnet(x)
        # x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.view(-1)


# Here we define our model as a class
class LeNet5(nn.Module):
    """
    Input - 1x32x32
    C1 - 6@28x28 (5x5 kernel)
    tanh
    S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
    C3 - 16@10x10 (5x5 kernel, complicated shit)
    tanh
    S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
    C5 - 120@1x1 (5x5 kernel)
    F6 - 84
    tanh
    F7 - 10 (Output)
    """
    def __init__(self):
        super(LeNet5, self).__init__()

        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv1d(3, 18, kernel_size=5)),
            ('relu1', nn.ReLU()),
            ('s2', nn.MaxPool1d(kernel_size=2, stride=2)),
            ('c3', nn.Conv1d(6, 16, kernel_size=(5, 5))),
            ('relu3', nn.ReLU()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c5', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('relu5', nn.ReLU())
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(120, 84)),
            ('relu6', nn.ReLU()),
            ('f7', nn.Linear(84, 10)),
            ('sig7', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output