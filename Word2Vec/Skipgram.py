# implement Skipgram
# windowは調整可能


import torch
import torch.nn as nn
import torch.nn.functional as F
from Preprocess import pad_seq
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD = 0
UNK = 1
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'


class DataloaderSG(object):
    def __init__(self, text, batch_size, window):
        self.text = text
        self.batch_size = batch_size
        self.window = window
        self.s_pointer = 0
        self.w_poiner = 0
        self.max_s_poiner = len(text)

    def __iter__(self):
        return self

    def __next__(self):
        batch_X = []
        batch_Y = []
        while len(batch_X) <= self.batch_size:
            sen = self.text[self.s_pointer]
            # input words
            word_X = sen[self.w_poiner]
            # output words
            start = max(self.w_poiner - self.window, 0)
            word_Y = sen[start:self.w_poiner] + \
                sen[self.w_pointer + 1:self.w_poiner + self.window + 1]
            word_Y = pad_seq(word_Y, self.window * 2)
            batch_X.append(word_X)
            batch_Y.append(word_Y)
            self.w_poiner += 1

            if self.w_poiner >= len(sen):
                self.w_poiner = 0
                self.s_pointer += 1

                if self.s_pointer >= self.max_s_poiner:
                    self.s_pointer = 0
                    raise StopIteration
        batch_X = torch.tensor(word_X, dtype=torch.long, device=device)
        batch_Y = torch.tensor(word_Y, dtype=torch.long, device=device)

        return batch_X, batch_Y


class Skipgram(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(Skipgram, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        # 埋め込み
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        # 全結合
        self.linear = nn.Linear(self.embedding_size, self.vocab_size, bias=False)

    def forward(self, batch_X, batch_Y):
        emb_X = self.embedding(batch_X)
        lin_X = self.linear(emb_X)
        log_prob_X = F.log_softmax(lin_X, dim=-1)
        log_prob_X = torch.gather(log_prob_X, 1, batch_Y)
        # マスキング
        log_prob_X = log_prob_X * (batch_Y != PAD).float()
        loss = log_prob_X.sum(1).mean().neg()
        return loss
