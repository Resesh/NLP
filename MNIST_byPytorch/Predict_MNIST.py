#モジュールのインポート

import sys, os
sys.path.append('../')
from Pytorch_MLP import MLP
import Get_MNIST
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

#モデリング
#MLPで実装

mlp = MLP(in_dim, hid_dim, out_dim)

optimizer = optim.SGD(mlp.parameters(),
    lr = lr)

criterion = nn.NLLLoss()


