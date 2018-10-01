#implement CBOW(Continuous Bug Of Words)

#setting hypermparameters


batch_size = 64
n_batches = 500
vocab_size = len(vocab.word2id)
embedding_size = 300

"batch_size: ミニバッチのサイズ"
"n_batches: 今回学習するミニバッチの数"
"vocab_size: 語彙の総数"
"embedding_size: 各単語の次元数"

