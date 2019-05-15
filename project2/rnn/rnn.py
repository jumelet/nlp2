
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from torch.autograd import Variable


class LM(nn.Module):
    def __init__(self,
                 batch_size: int,
                 vocabulary_size: int,
                 model_type: str = 'LSTM',
                 input_emb_dim: int = 100,
                 hidden_size: int = 100,
                 lstm_num_layers: int = 2,
                 dropout: float = 0.,
                 device: str = 'cuda:0') -> None:

        super(LM, self).__init__()

        self.lstm_num_layers = lstm_num_layers
        self.lstm_num_hidden = hidden_size
        self.batch_size = batch_size
        self.input_emb_dim = input_emb_dim

        rnn = {
            'LSTM': nn.LSTM,
            'GRU': nn.GRU,
        }[model_type]

        self.rnn = rnn(
            input_emb_dim,
            hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout,
        ).to(device)

        self.encoder = nn.Embedding(vocabulary_size, input_emb_dim).to(device)
        self.decoder = nn.Linear(hidden_size, vocabulary_size).to(device)

        self.hidden = self.init_hidden()

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, hidden=None):
        encoded = self.encoder(x)

        if hidden is None:
            hidden = self.hidden

        x= self.rnn(encoded, hidden)
        out = self.decoder(h)

        return out, hidden

    def init_hidden(self):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.lstm_num_layers, self.batch_size, self.lstm_num_hidden).zero_()),
                Variable(weight.new(self.lstm_num_layers, self.batch_size, self.lstm_num_hidden).zero_()))

    def reset_hidden(self):
        self.hidden = tuple(Variable(h.data) for h in self.hidden)
