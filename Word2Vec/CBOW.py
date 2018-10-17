# implement CBOW
# set hyperparameters
# settings token

import torch
import Preprocess
from Preprocess import pad_seq
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD = 0
UNK = 1
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'

# create dictionary

word2id = {
    PAD_TOKEN: PAD,
    UNK_TOKEN: UNK,
}
vocab = Preprocess.Vocab(word2id=word2id)
batch_size = 64
n_batches = 500
vocab_size = len(vocab.word2id)
embedding_size = 300

"batch_size: ミニバッチのサイズ"
"n_batches: 今回学習するミニバッチの数"
"vocab_size: 語彙の総数"
"embedding_size: 各単語の次元数"

# preapare_classes


class TestIter(object):
    def __init__(self):
        self.iter = 0
        self.max_iter = 5

    def __iter__(self):
        print("iter関数が呼び出されました")
        return self

    def __next__(self):
        self.iter += 1
        print("next関数が呼び出されました({}回目)".format(self.iter))
        if self.iter < self.max_iter:
            return None
        else:
            print("max_iterに達したので終了します")
            raise StopIteration


class DataloaderCBOW(object):
    def __init__(self, text, batch_size, window):
        self.text = text
        self.batch_size = batch_size
        self.window = window
        self.s_pointer = 0
        self.w_pointer = 0
        self.max_s_pointer = len(text)

    def __iter__(self):
        return self

    def __next__(self):
        batch_X = []
        batch_Y = []
        # 各バッチごとの操作
        while len(batch_X) < self.batch_size:
            # 走査するsentence
            sen = self.text[self.s_pointer]
            # 予測単語について
            word_Y = self.text[self.w_pointer]
            # 入力単語について
            # w_pointerは外す
            start = max(0, self.w_pointer - self.window)
            word_X = sen[start:self.w_pointer] + \
                sen[self.w_pointer + 1:self.w_pointer + self.window + 1]
            word_X = pad_seq(word_X, self.window*2)
            batch_X.append(word_X)
            batch_Y.append(word_Y)
            self.w_pointer += 1

            if self.w_pointer >= len(sen):
                self.w_pointer = 0
                self.s_pointer += 1
                if self.s_pointer >= self.max_s_pointer:
                    self.s_pointer = 0
                    raise StopIteration
        batch_X = torch.tensor(word_X, dtype=torch.long, device=device)
        batch_Y = torch.tensor(word_Y, dtype=torch.long, device=device)

        return batch_X, batch_Y
