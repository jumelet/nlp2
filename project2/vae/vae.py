"""
Sentence VAE
------------
Samuel R. Bowman et al., 2015, Generating Sentences from a Continuous Space, SIGNLL

Taking inspiration from https://github.com/pytorch/examples/blob/master/vae/main.py
"""

from __future__ import print_function
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNEncoder(nn.Module):
    def __init__(self, rnn_type, nlayers, bidirectional, embedding, edim, hdim, zdim, vocab_len,
                 batch_size, dropout, device):
        super(RNNEncoder, self).__init__()
        self.V = vocab_len
        self.batch_size = batch_size
        ndirections = 2 if bidirectional else 1
        self.nlayers = nlayers * ndirections
        self.hdim = hdim
        self.device = device

        if rnn_type in ['LSTM', 'GRU']:
            self.rnn_type = rnn_type
            self.rnn = getattr(nn, rnn_type)(edim, hdim, nlayers,
                                             bidirectional=bidirectional,
                                             dropout=dropout,
                                             batch_first=True).to(device)
        else:
            raise ValueError("""An invalid option for `--model` was supplied,
                                options are ['LSTM', 'GRU']""")

        self.embed = embedding
        self.encode = nn.Linear(hdim * ndirections, zdim).to(device)
        self.init_weights()
        self.hidden = None
        self.reset_hidden()

    def init_weights(self):
        initrange = 0.1
        # self.embed.weight.data.uniform_(-initrange, initrange)
        self.encode.bias.data.fill_(0)
        self.encode.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_, lengths=None, hidden=None):
        embed_sequence = self.embed(input_)

        if lengths is not None:
            embed_sequence = pack_padded_sequence(embed_sequence,
                                                  lengths=lengths,
                                                  batch_first=True)
        if hidden is None:
            hidden = self.hidden

        output, _ = self.rnn(embed_sequence, hidden)
        last_output = output[:, -1, :]

        if lengths is not None:
            last_output = pad_packed_sequence(last_output, batch_first=True)[0]

        return self.encode(last_output)

    def reset_hidden(self, bsz=None):
        if bsz is None:
            bsz = self.batch_size
        if self.rnn_type == 'GRU':
            self.hidden = torch.zeros(self.nlayers, bsz, self.hdim).to(self.device)
        else:
            self.hidden = (torch.zeros(self.nlayers, bsz, self.hdim).to(self.device),
                           torch.zeros(self.nlayers, bsz, self.hdim).to(self.device))


class RNNDecoder(nn.Module):
    def __init__(self, rnn_type, nlayers, bidirectional, embedding, edim, hdim, zdim, vocab_len,
                 dropout, device):
        super(RNNDecoder, self).__init__()

        ndirections = 2 if bidirectional else 1
        self.V = vocab_len
        self.hdim = hdim
        self.nlayers = nlayers * ndirections
        self.device = device

        if rnn_type in ['LSTM', 'GRU']:
            self.rnn_type = rnn_type
            self.rnn = getattr(nn, rnn_type)(edim, hdim, nlayers,
                                             bidirectional=bidirectional,
                                             dropout=dropout,
                                             batch_first=True).to(device)
        else:
            raise ValueError("""An invalid option for `--model` was supplied,
                                options are ['LSTM', 'GRU']""")

        # self.embed = nn.Embedding(self.V, edim, padding_idx=0).to(device)
        self.embed = embedding
        self.decode = nn.Linear(zdim, self.hdim * self.nlayers).to(device)
        self.tovocab = nn.Linear(self.hdim * ndirections, self.V).to(device)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        # self.embed.weight.data.uniform_(-initrange, initrange)
        self.decode.bias.data.fill_(0)
        self.decode.weight.data.uniform_(-initrange, initrange)
        self.tovocab.bias.data.fill_(0)
        self.tovocab.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, z):
        if self.rnn_type == 'GRU':
            h0 = self.decode(z).view(-1, input.size(0), self.hdim)
        else:
            h0 = self.decode(z).view(-1, input.size(0), self.hdim)
            h0 = (h0, torch.zeros_like(h0, device=self.device))
        output, _ = self.rnn(self.embed(input), h0)

        log_p = F.log_softmax(self.tovocab(output), dim=-1)

        return log_p


class SentenceVAE(nn.Module):
    def __init__(self,
                 batch_size,
                 rnn_type,
                 nlayers,
                 bidirectional,
                 edim,
                 hdim,
                 zdim,
                 vocab_len,
                 word_dropout_prob=0.,
                 rnn_dropout=0.,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(SentenceVAE, self).__init__()

        self.embedding = nn.Embedding(vocab_len, edim, padding_idx=0).to(device)
        self.encoder = RNNEncoder(rnn_type, nlayers, bidirectional, self.embedding, edim, hdim,
                                  zdim, vocab_len, batch_size, rnn_dropout, device)
        self.decoder = RNNDecoder(rnn_type, nlayers, bidirectional, self.embedding, edim, hdim,
                                  zdim, vocab_len, rnn_dropout, device)
        self.project_loc = nn.Linear(zdim, zdim).to(device)
        self.project_logv = nn.Linear(zdim, zdim).to(device)
        self.device = device
        self.dropout_prob = word_dropout_prob
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.project_loc.bias.data.fill_(0)
        self.project_loc.weight.data.uniform_(-initrange, initrange)
        self.project_logv.bias.data.fill_(0)
        self.project_logv.weight.data.uniform_(-initrange, initrange)

    def encode(self, input_):
        h = self.encoder(input_)
        return self.project_loc(h), F.softplus(self.project_logv(h))

    @staticmethod
    def reparametrize(loc, logv):
        std = torch.exp(0.5 * logv)
        eps = torch.randn_like(std)
        return loc + eps * std  # z

    def decode(self, input_, z):
        """ Randomly replace decoder input with <unk> """
        if self.training and self.dropout_prob > 0:
            mask = torch.rand(input_.size(), device=self.device)
            mask[mask < self.dropout_prob] = 0
            mask[mask >= self.dropout_prob] = 1
            if mask.size(0) > 1:
                mask[:, 0] = 1  # keep begin of sentence
            input_ = torch.mul(input_, mask.long())
        return self.decoder(input_, z)

    def forward(self, input_):
        loc, logv = self.encode(input_)
        z = self.reparametrize(loc, logv)
        log_p = self.decode(input_, z)
        return log_p, loc, logv
