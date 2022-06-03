import torch
import torch.nn as nn
import torch.functional as F
import numpy as np

class MLPModel(nn.Module):
    def __init__(self, linear_layer_size=101*40, hid_dim1=600, hid_dim2=100, dropout_rate=0.5,filter_sizes=None):
        super().__init__()
        print(f"training with dropout={dropout_rate}")
        self.input_dim = linear_layer_size
        self.hid_dim1 = hid_dim1
        self.hid_dim2 = hid_dim2
        self.dropout = nn.Dropout(dropout_rate)
        self.linear1 = nn.Linear(self.input_dim, hid_dim1)
        self.linear2 = nn.Linear(hid_dim1, hid_dim2)
        self.linear3 = nn.Linear(hid_dim2, 1)
        self.bn1 = nn.BatchNorm1d(num_features=hid_dim1)
        self.bn2 = nn.BatchNorm1d(num_features=hid_dim2)
        
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = np.inf
        
    def forward(self, src):
        src = src.view((-1,self.input_dim))
        hidden1 = self.linear1(src)
        hidden1 = self.bn1(hidden1)
        hidden1 = self.dropout(hidden1)
        hidden1 = F.relu(hidden1)
        
        hidden2 = self.linear2(hidden1)
        hidden2 = self.bn2(hidden2)
        hidden2 = self.dropout(hidden2)
        hidden2 = F.relu(hidden2)
        output = self.linear3(hidden2)
        output = torch.sigmoid(output)
        return output
    
    def set_device(self, device):
        self.to(device)