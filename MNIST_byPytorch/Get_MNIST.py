#pytorchのモジュールを利用する場合

import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
from sklearn.metrics import f1_score


# ハイパーパラメータの設定

in_dim  = 784
hid_dim = 200
out_dim = 10
lr = 0.001
batch_size = 32
num_epochs = 40



train_data = torchvision.datasets.FashionMNIST(
    './data/fashion-mnist', train = True,
    transform = torchvision.transforms.ToTensor(), download = True
    )

test_data = torchvision.datasets.FashionMNIST(
    './data/fashion-mnist', train = True, 
    transform = torchvision.transforms.ToTensor(), download = True
)

#define DataLoader

train_data_loader = torch.utils.data.DataLoader(
    dataset = train_data,
    batch_size = batch_size,
    shuffle = True
)

test_data_loader = torch.utils.data.DataLoader(
    dataset = test_data,
    batch_size = batch_size,
    shuffle = True
)
