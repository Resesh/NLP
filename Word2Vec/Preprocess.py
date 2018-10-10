# define functions implemented in preprocessing
# how can I est Module 'Mecab' ??
# line2 is already resolved
import MeCab
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tagger = MeCab.tagger("-Ochasen")
node = tagger.parse("坊主が屏風に上手に坊主の絵を描いた")
print(node)

# preprocess がなぜ実行できないのか
# No module named 'torch'
# python pathが通ってないのはなぜか
# 通ったよ(user settings とworkspace settingsの違い)

# settings token

PAD = 0
UNK = 1
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'

# create dictionary

word2id = {
    PAD_TOKEN: PAD,
    UNK_TOKEN: UNK,
}

# min_countは調整するパラメータで、
# 何回以上現れた単語を辞書にするかを決めている


class Vocab(object):
    def __init__(self, word2id={}):
        self.word2id = dict(word2id)
        self.id2word = {v: k for k, v in self.word2id.items()}

    def build_vocab(self, sentences, min_count):
        # 辞書作成
        word_counter = {}
        for sentence in sentences:
            for word in sentence:
                word_counter[word] = word_counter.get(word, 0) + 1

        for word, count in sorted(word_counter.items(), key=lambda x: -x[1]):
            if count < min_count:
                break
            _id = len(self.word2id)
            self.word2id.setdefault(word, _id)
            self.id2word[_id] = word


def load_data():

    with open('./data/text8') as f:
        line = f.readline()
        line = line.strip.split()
    return line


text = load_data()
# どこまで取り出すかは調整
text = text[:1000000]


vocab = Vocab(word2id=word2id)
vocab.build_vocab([text], min_count=3)
print("語彙数:", len(vocab.word2id))
print(len(text))
