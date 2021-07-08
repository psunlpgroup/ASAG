import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LRClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_size, embedding_matrix):
        super(LRClassifier, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # self.embedding = nn.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix])
        # print("Torch.float tensor: ", torch.FloatTensor(embedding_matrix))
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix))

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2out = nn.Linear(hidden_dim, output_size)

    # self.softmax = nn.LogSoftmax()



    def init_hidden(self, batch_size):
        return (autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)),
                autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)))

    def forward(self, batch, lengths):
        self.hidden = self.init_hidden(batch.size(-1))

        # print('batch: ', batch.size())
        # print('lengths: ', lengths.squeeze())
        embeds = self.embedding(batch)
        # print('embeds: ', embeds.size())
        # [1, 50, 100]
        # pack LSTM input
        # why? see this link: https://gist.github.com/HarshTrivedi/f4e7293e941b17d19058f6fb90ab0fec
        packed_input = pack_padded_sequence(embeds, lengths, batch_first=True)
        # print("packed_input.data.size: ", packed_input.data.size())
        # print("packed_input.batch_sizes.size: ", packed_input.batch_sizes.size())

        outputs, (ht, ct) = self.lstm(packed_input)
        # use hidden state ht as result
        # print("ht: ", ht.size())
        output = self.hidden2out(ht.view(-1, self.hidden_dim))
        # print("output after dropout: ", output)
        # print("output after softmax: ", output.size())

        return output