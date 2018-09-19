#pytorchのモジュールを利用する場合

import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms, models

def get_mnist(train_data, test_data):
    
    train_data = torchvision.datasets.FashionMNIST(
    './data/fashion-mnist', train = True,
    transform = torchvision.transforms.ToTensor(), download = True
    )

    test_data = torchvision.datasets.FashionMNIST(
    './data/fashion-mnist', train = True, 
    transform = torchvision.transforms.ToTensor(), download = True
)
    return train_data, test_data


