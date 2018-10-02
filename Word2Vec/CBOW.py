#implement CBOW


#set hyperparameters

batch_size = 64
n_batches = 500
vocab_size = len(vocab.word2id)
embedding_size = 300

"batch_size: ミニバッチのサイズ"
"n_batches: 今回学習するミニバッチの数"
"vocab_size: 語彙の総数"
"embedding_size: 各単語の次元数"

#preapare_classes


class TestIter(object):
    def __init__(self):
        self.iter = 0
        self.max_iter = 5
    
    def __iter__(self): # 必須
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


class Dataloader(self, text, batch_size, window):
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
        #各バッチごとの操作
        while len(batch_X) < self.batch_size:
            #走査するsentence
            sen = self.text[self.s_pointer]
            #予測単語について
            word_Y = self.text[self.w_pointer]
            #入力単語について
            #w_pointerは外す
            start = max(0, self.w_pointer - self.window)
            word_X = sen[start:self.w_pointer] + \
            sen[self.w_pointer + 1:self.w_pointer + self.window + 1]
            


            