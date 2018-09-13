import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
from sklearn.metrics import f1_score

class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(in_dim, hid_dim)
        self.linear2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.log_softmax(self.linear2(x), dim = -1)
        return x
