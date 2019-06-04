import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LM(nn.Module):
    def __init__(self,
                 batch_size: int,
                 vocabulary_size: int,
                 model_type: str = 'LSTM',
                 input_emb_dim: int = 100,
                 hidden_size: int = 100,
                 lstm_num_layers: int = 2,
                 dropout: float = 0.,
                 tie_weights: bool = False,
                 device: str = 'cuda:0') -> None:

        super(LM, self).__init__()

        self.lstm_num_layers = lstm_num_layers
        self.lstm_num_hidden = hidden_size
        self.batch_size = batch_size
        self.input_emb_dim = input_emb_dim

        self.device = device

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
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(hidden_size, vocabulary_size).to(device)

        if tie_weights:
            self.decoder.weight = self.encoder.weight

        self.hidden = self.init_hidden()

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_, lengths=None, hidden=None):
        encoded = self.encoder(input_.to(self.device))

        if lengths is not None:
            encoded = pack_padded_sequence(encoded,
                                           lengths=lengths,
                                           batch_first=True)

        if hidden is None:
            hidden = self.hidden

        final_h, new_hidden = self.rnn(encoded, hidden)
        if hidden is None:
            self.hidden = new_hidden

        if lengths is not None:
            final_h = pad_packed_sequence(final_h, batch_first=True)[0]
        out = self.decoder(self.dropout(final_h))

        return out, new_hidden

    def init_hidden(self, bsz=None):
        if bsz is None:
            bsz = self.batch_size
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.lstm_num_layers, bsz, self.lstm_num_hidden).zero_()),
                Variable(weight.new(self.lstm_num_layers, bsz, self.lstm_num_hidden).zero_()))

    def reset_hidden(self):
        self.hidden = tuple(Variable(h.data) for h in self.hidden)
