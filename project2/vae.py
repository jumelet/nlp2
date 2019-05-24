"""
Sentence VAE
------------
Samuel R. Bowman et al., 2015, Generating Sentences from a Continuous Space, SIGNLL

Taking inspiration from https://github.com/pytorch/examples/blob/master/vae/main.py
"""

from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from nltk.tree import Tree
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from torch.distributions.multivariate_normal import MultivariateNormal
from scipy.special import logsumexp
import numpy as np

parser = argparse.ArgumentParser(description='Sentence VAE')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--train', type=str, default='data/23.auto.clean',  #default='data/02-21.10way.clean',
                    help='file path of the training data')
parser.add_argument('--valid', type=str, default='data/22.auto.clean',
                    help='file path of the validation data')
parser.add_argument('--test', type=str, default='data/23.auto.clean',
                    help='file path of the test data')
parser.add_argument('--rnn', type=str, default='GRU', metavar='N',
                    choices=['GRU', 'LSTM'],
                    help='type of RNN (default: GRU)')
parser.add_argument('--nlayers', type=int, default=1, metavar='N',
                    help='number of RNN layers (default: 1)')
parser.add_argument('--bidir', action='store_true', default=True,
                    help='enables RNN bidirectionality')
parser.add_argument('--edim', type=int, default=353, metavar='N',
                    help='embedding dimensions (default: 353)')
parser.add_argument('--hdim', type=int, default=191, metavar='N',
                    help='hidden RNN dimensions (default: 191)')
parser.add_argument('--zdim', type=int, default=13, metavar='N',
                    help='number of latent codes (default: 13)')
parser.add_argument('--word_dropout', type=float, default=0.4,
                    help='dropout probability for an input token (default: 0.4)')
parser.add_argument('--isamples', type=int, default=5, metavar='N',
                    help='number of importance samples (default: 5)')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
torch.manual_seed(args.seed)


EOS = '[EOS]'
BOS = '[BOS]'
UNK = '[UNK]'


def batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


class Split(object):
    def __init__(self, sentences, w2i, n=None):
        self.sentences = sentences
        self.w2i = w2i
        self.V = len(w2i)
        if n is None:
            self.length = len(sentences)
        else:
            self.length = n

    def data(self, batch_size):
        for batch in batches(self.sentences, batch_size):
            input_batch = [torch.tensor([self.w2i[token] for token in sentence[:-1]]) for sentence in batch]
            target_batch = [torch.tensor([self.w2i[token] for token in sentence[1:]]) for sentence in batch]
            input_batch = pad_sequence(input_batch)
            target_batch = pad_sequence(target_batch)
            # onehot = torch.Tensor(torch.zeros(batch.size(0), self.V).scatter_(1, batch, 1.), type=torch.LongTensor)
            yield input_batch, target_batch

    def __len__(self):
        return self.length


class Corpus(object):
    def __init__(self, train_path, validation_path, test_path):
        counter = Counter()
        train_sentences, train_len, counter = self.load(train_path, counter)
        valid_sentences, valid_len, counter = self.load(validation_path, counter)
        test_sentences, test_len, counter = self.load(test_path, counter)

        self.vocab = [UNK, EOS, BOS] + [w for (w, freq) in counter.most_common()]
        self.w2i = {w: i for (i, w) in enumerate(self.vocab)}
        self.i2w = {i: w for (w, i) in self.w2i.items()}

        self.training = Split(train_sentences, self.w2i, train_len)
        self.test = Split(test_sentences, self.w2i, test_len)
        self.validation = Split(valid_sentences, self.w2i, valid_len)

    def load(self, path, counter=None):
        if counter is None:
            counter = Counter()

        print(f'Reading corpus: {path}')
        sentences, len = [], 0
        with open(path, 'r') as f:
            for line in f.readlines():
                tree = Tree.fromstring(line)
                sentence = tree.leaves()
                counter += Counter(sentence)
                sentence = [BOS] + sentence + [EOS]
                sentences.append(sentence)
                len += 1
        print(f'Done. {len} sentences.')
        # print(f'Most frequent words: {counter.most_common(10)}\n')

        return sentences, len, counter

    def tokens_to_ids(self, seq):
        return [self.w2i[w] for w in seq]

    def ids_to_tokens(self, seq):
        return [self.i2w[i] for i in seq]


class Annealing(object):
    def __init__(self, type='linear', nsteps=5000):
        self.nsteps = nsteps
        self.step = 0

        if type not in ['linear', 'sigmoid']:
            raise ValueError('Invalid annealing type: {}'.format(type))
        self.type = type

    def rate(self):
        self.step += 1
        if self.type == 'linear':
            return self.step / self.nsteps
        else:
            raise NotImplementedError()

    def reset(self):
        self.step = 0


class RNNEncoder(nn.Module):
    def __init__(self, rnn_type, nlayers, bidirectional, edim, hdim, zdim, vocab):
        super(RNNEncoder, self).__init__()
        self.V = len(vocab)

        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(edim, hdim, nlayers, bidirectional=bidirectional)
        else:
            raise ValueError("""An invalid option for `--model` was supplied,
                                options are ['LSTM', 'GRU']""")

        if bidirectional:
            packed_hdim = 2 * hdim
        else:
            packed_hdim = hdim

        self.embed = nn.Embedding(self.V, edim, padding_idx=0)
        self.encode = nn.Linear(packed_hdim, zdim)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        output, hidden = self.rnn(self.embed(input), hidden)
        output = output[-1, :, :]
        return self.encode(output)


class RNNDecoder(nn.Module):
    def __init__(self, rnn_type, nlayers, bidirectional, edim, hdim, zdim, vocab):
        super(RNNDecoder, self).__init__()

        self.V = len(vocab)
        self.hdim = hdim

        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(edim, hdim, nlayers, bidirectional=bidirectional)
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

    def forward(self, input, z):
        h0 = self.decode(z).view(-1, input.size(1), self.hdim)
        output, _ = self.rnn(self.embed(input), h0)
        log_p = F.log_softmax(self.tovocab(output), dim=-1)
        return log_p


class VAE(nn.Module):
    def __init__(self, batch_size, rnn_type, nlayers, bidirectional, edim, hdim, zdim, vocab, word_dropout_prob=0.):
        super(VAE, self).__init__()

        self.encoder = RNNEncoder(rnn_type, nlayers, bidirectional, edim, hdim, zdim, vocab)
        self.decoder = RNNDecoder(rnn_type, nlayers, bidirectional, edim, hdim, zdim, vocab)
        self.project_loc = nn.Linear(zdim, zdim)
        self.project_scale = nn.Linear(zdim, zdim)

        self.nlayers = nlayers
        self.zdim = zdim
        self.bidirectional = bidirectional
        self.hdim = hdim
        self.dropout_prob = word_dropout_prob
        self.hidden = torch.Tensor(torch.zeros(self.nlayers + int(self.bidirectional) * self.nlayers,
                                               batch_size,
                                               self.hdim))

    def encode(self, input, hidden=None):
        if hidden is None:
            hidden = self.hidden
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


def word_prediction_accuracy(logp, target):
    argmax = torch.argmax(logp, dim=1)
    return torch.mean(torch.eq(argmax, target).float())


def KLLoss(loc, scale, annealing=None):
    kl_loss = -0.5 * torch.sum(1 + (scale ** 2) - (loc ** 2) - (scale ** 2).exp())
    if annealing:
        kl_loss *= annealing.rate()
    return kl_loss


def loss_function(logp, target, loc, scale, annealing=None):
    NLL = torch.nn.NLLLoss(ignore_index=0)
    nll_loss = NLL(logp, target)
    kl_loss = KLLoss(loc, scale, annealing)
    return nll_loss + kl_loss


def approximate_sentence_NLL(model, loc, scale, sent, target, nsamples=16):
    """
        NLL with Importance Sampling
    """
    encoder_distribution = MultivariateNormal(loc, torch.diag((scale ** 2).squeeze(0)))
    prior_distribution = MultivariateNormal(torch.tensor([0.] * loc.size(-1)), torch.eye(loc.size(-1)))

    NLL = torch.nn.NLLLoss(ignore_index=0, reduction='sum')
    samples = []
    for s in range(nsamples):
        z = encoder_distribution.sample((1,))                  # sampling a z
        log_q_z_x = encoder_distribution.log_prob(z)           # the probablity of z under the encoder distribution
        log_p_z = prior_distribution.log_prob(z)               # the probability of z under a gaussian prior
        logp = model.decode(sent, z)
        log_p_x_z = - NLL(logp.squeeze(1), target.squeeze(1))  # the sentence probability given the latent variable

        samples.append(log_p_x_z.item() + log_p_z.item() - log_q_z_x.item())
    return np.log(nsamples) - logsumexp(samples)


def train(model, optimizer, train_split, batch_size, epoch):
    model.train()
    train_loss = 0
    wpa = 0
    n_batches = 0
    annealing = Annealing('linear', 2000)
    for batch_idx, (data, target) in enumerate(tqdm(train_split.data(batch_size), total=len(train_split) // batch_size)):

        n_batches += 1
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        log_p, loc, scale = model(data)

        loss = loss_function(log_p,
                             target,
                             loc,
                             scale,
                             annealing)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        wpa += word_prediction_accuracy(log_p, target)

    train_loss /= len(train_split)
    wpa /= n_batches
    print('====> Epoch: {} Average training loss: {:.4f}  WPA: {:.4f}'.format(epoch, train_loss, wpa))
    return train_loss, wpa


def validate(model, valid_split, batch_size, epoch):
    model.eval()
    valid_loss = 0
    wpa = 0
    n_batches = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(valid_split.data(batch_size)):
            n_batches += 1
            data = data.to(device)
            target = target.to(device)

            log_p, loc, scale = model(data)
            loss = loss_function(log_p,
                                 target,
                                 loc,
                                 scale)
            valid_loss += loss.item()
            wpa += word_prediction_accuracy(log_p, target)

    valid_loss /= len(valid_split)
    wpa /= n_batches
    print('====> Epoch: {} Average validation loss: {:.4f}  WPA: {:.4f}'.format(epoch, valid_loss, wpa))
    return valid_loss, wpa


def test(model, test_split, nsamples=16):
    model.eval()
    test_loss = 0
    wpa = 0
    with torch.no_grad():
        for (data, target) in test_split.data(1):
            data = data.to(device)
            target = target.to(device)
            logp, loc, scale = model(data)
            test_loss += approximate_sentence_NLL(model, loc, scale, data, target, nsamples)
            wpa += word_prediction_accuracy(logp, target)

    test_loss /= len(test_split)
    wpa /= len(test_split)
    print('====> Test set loss: {:.4f}WPA: {:.4f}'.format(test_loss, wpa))
    return test_loss, wpa


if __name__ == "__main__":

    corpus = Corpus(args.train, args.valid, args.test)
    print('Vocabulary size:', len(corpus.vocab))

    model = VAE(args.batch_size,
                rnn_type=args.rnn,
                nlayers=args.nlayers,
                bidirectional=args.bidir,
                edim=args.edim,
                hdim=args.hdim,
                zdim=args.zdim,
                vocab=corpus.vocab,
                word_dropout_prob=args.word_dropout)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_stats = []
    valid_stats = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_wpa = train(model, optimizer, corpus.training, args.batch_size, epoch)
        valid_loss, valid_wpa = validate(model, corpus.validation, args.batch_size, epoch)

        train_stats.append((train_loss, train_wpa))
        valid_stats.append((valid_loss, valid_wpa))
        test(model, corpus.test, nsamples=args.isamples)


##########################################################################################
# self.eos = self.embed(torch.tensor(batch_size * [self.w2i['[EOS]']]))
#
# seq = []
# log_p = []
# for i in range(len):
#     # print()
#     # print('!!!!', self.eos.view(1, batch_size, -1).shape)
#     # print()
#     output, hidden = self.rnn(input,
#                               hidden)
#     log_p_output = F.log_softmax(self.tovocab(output))
#     next_token_id = torch.argmax(log_p_output, dim=-1)
#     input = self.embed(next_token_id)
#     log_p.append(log_p_output)
#     seq.append(next_token_id)
# return torch.stack(log_p), torch.stack(seq)
