#モジュールのインポート

import MLP_Pytorch
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
from Get_MNIST import get_mnist


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ハイパーパラメータの設定

in_dim  = 784
hid_dim = 200
out_dim = 10
lr = 0.001
batch_size = 32
num_epochs = 40

#define DataLoader
#train, test = get_mnist(train_data, test_data)

train_data = torchvision.datasets.FashionMNIST(
    './data/fashion-mnist',
    transform=torchvision.transforms.ToTensor(),
    train=True,
    download=True)

test_data = torchvision.datasets.FashionMNIST(
    './data/fashion-mnist',
    transform=torchvision.transforms.ToTensor(),
    train=False,
    download=True)

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



#モデリング
#MLPで実装

mlp = MLP_Pytorch.MLP(in_dim, hid_dim, out_dim)

optimizer = optim.SGD(mlp.parameters(), lr = lr)
criterion = nn.NLLLoss()

for epoch in range(num_epochs + 1):
    losses_train = []
    losses_test = []
    preds_train = []
    preds_test = []
    trues_train = []
    trues_test = []

    mlp.train()

    for x, t in train_data_loader:
        true = t.tolist()
        trues_train.extend(true)

        #勾配初期化
        mlp.zero_grad()

        #onto GPU

        x = x.to(device)
        x = x.view(x.size(0), -1)
        t = t.to(device)

         #順伝播
        y = mlp.forward(x)

        #誤差計算
        loss = criterion(y, t)

        # 逆伝播
        loss.backward()

        #パラメータ更新
        optimizer.step()

        #出力を格納
        pred = y.argmax(1).tolist()
        preds_train.extend(pred)

        losses_train.append(loss.tolist())

    #evaluation
    mlp.eval()

    preds = []

    for x, t in test_data_loader:
        true = t.tolist()
        trues_test.extend(true)

        x = x.to(device)
        x = x.vieww(x.size(0), -1)
        t = t.to(device)

        y = mlp.forward(x)

        loss = criterion(y, t)

        pred = y.argmax(1).tolist()
        preds += pred
        preds_test.extend(pred)

        losses_test.append(loss.tolist())


    print('EPOCH: {}, Train [Loss: {:.3f}, F1: {:.3f}], Valid [Loss: {:.3f}, F1: {:.3f}]'.format(
        epoch,
        np.mean(losses_train),
        f1_score(trues_train, preds_train, average='macro'),
        np.mean(losses_test),
        f1_score(trues_test, preds_test, average='macro')
    ))

# 予測結果を保存
    SUBMISSION_PATH = 'submission.csv'
    with open(SUBMISSION_PATH, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
    writer.writerow(preds)




