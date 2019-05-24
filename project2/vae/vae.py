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
    def __init__(self, rnn_type, nlayers, bidirectional, edim, hdim, zdim, vocab_len, batch_size):
        super(RNNEncoder, self).__init__()
        self.V = vocab_len
        self.batch_size = batch_size
        self.nlayers = nlayers + int(bidirectional) * nlayers
        self.hdim = hdim

        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(edim, hdim, nlayers, bidirectional=bidirectional, batch_first=True)
        else:
            raise ValueError("""An invalid option for `--model` was supplied,
                                options are ['LSTM', 'GRU']""")


        self.embed = nn.Embedding(self.V, edim, padding_idx=0)
        self.encode = nn.Linear(hdim + int(bidirectional) * hdim, zdim)
        self.init_weights()
        self.reset_hidden()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.encode.bias.data.fill_(0)
        self.encode.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, lengths=None, hidden=None):
        embed_sequence = self.embed(input)

        if lengths is not None:
            embed_sequence = pack_padded_sequence(embed_sequence,
                                           lengths=lengths,
                                           batch_first=True)
        if hidden is None:
            hidden = self.hidden

        output, hidden = self.rnn(embed_sequence, hidden)
        final_h = output[:, -1, :]

        if lengths is not None:
            final_h = pad_packed_sequence(final_h, batch_first=True)[0]

        return self.encode(final_h)

    def reset_hidden(self, bsz=None):
        if bsz is None:
            bsz = self.batch_size
        self.hidden = torch.zeros(self.nlayers, bsz, self.hdim)


class RNNDecoder(nn.Module):
    def __init__(self, rnn_type, nlayers, bidirectional, edim, hdim, zdim, vocab_len):
        super(RNNDecoder, self).__init__()

        self.V = vocab_len
        self.hdim = hdim

        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(edim, hdim, nlayers, bidirectional=bidirectional, batch_first=True)
        else:
            raise ValueError("""An invalid option for `--model` was supplied,
                                options are ['LSTM', 'GRU']""")

        if bidirectional:
            packed_hdim = 2 * hdim
        else:
            packed_hdim = hdim

        self.embed = nn.Embedding(self.V, edim, padding_idx=0)
        self.decode = nn.Linear(zdim, packed_hdim)
        self.tovocab = nn.Linear(packed_hdim, self.V)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.decode.bias.data.fill_(0)
        self.decode.weight.data.uniform_(-initrange, initrange)
        self.tovocab.bias.data.fill_(0)
        self.tovocab.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, z):
        h0 = self.decode(z).view(-1, input.size(0), self.hdim)
        output, _ = self.rnn(self.embed(input), h0)
        log_p = F.log_softmax(self.tovocab(output), dim=-1)
        return log_p


class SentenceVAE(nn.Module):
    def __init__(self, batch_size, rnn_type, nlayers, bidirectional, edim, hdim, zdim, vocab_len, word_dropout_prob=0.):
        super(SentenceVAE, self).__init__()

        self.encoder = RNNEncoder(rnn_type, nlayers, bidirectional, edim, hdim, zdim, vocab_len, batch_size)
        self.decoder = RNNDecoder(rnn_type, nlayers, bidirectional, edim, hdim, zdim, vocab_len)
        self.project_loc = nn.Linear(zdim, zdim)
        self.project_scale = nn.Linear(zdim, zdim)

        self.nlayers = nlayers
        self.zdim = zdim
        self.bidirectional = bidirectional
        self.hdim = hdim
        self.dropout_prob = word_dropout_prob

    def encode(self, input, hidden=None):
        h = self.encoder(input, hidden)
        return self.project_loc(h), F.softplus(self.project_scale(h))

    def reparametrize(self, loc, scale):
        std = torch.exp(0.5 * scale)
        eps = torch.randn_like(std)
        return loc + eps * std  # z

    def decode(self, input, z):
        # randomly replace decoder input with <unk>
        if self.dropout_prob > 0:
            mask = torch.rand(input.size())
            if torch.cuda.is_available():
                mask = mask.cuda()
            mask[mask < self.dropout_prob] = 0
            mask[mask >= self.dropout_prob] = 1
            mask[0, :] = 1  # always keep begin of sentence
            input = torch.mul(input, mask.long())
        return self.decoder(input, z)

    def forward(self, input):
        loc, scale = self.encode(input)
        z = self.reparametrize(loc, scale)
        log_p = self.decode(input, z)
        return log_p.permute(0, 2, 1), loc, scale
