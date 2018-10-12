# implement Skipgram
# windowは調整可能
class Skipgram(object):
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
            word_X = sen[w_poiner]
            # output words
            start = max(self.w_poiner - self.window, 0)
            word_Y = sen[start:self.w_poiner] + \
                sen[self.w_pointer + 1:self.w_poiner + self.window + 1]
            





