#define functions implemented in preprocessing

import Mecab
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tagger = Mecab.tagger("-Ochasen")
node = tagger.parse("坊主が屏風に上手に坊主の絵を描いた")
print(node)

