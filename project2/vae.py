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


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

EOS = '[EOS]'
BOS = '[BOS]'
UNK = '[UNK]'


def batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]


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
                # if len(sentence) > max_seq_len:
                #     max_seq_len = len(sentence)
                len += 1
        print(f'Done. {len} sentences.')
        # print(f'Most frequent words: {counter.most_common(10)}\n')

        return sentences, len, counter

    def tokens_to_ids(self, seq):
        return [self.w2i[w] for w in seq]

    def ids_to_tokens(self, seq):
        return [self.i2w[i] for i in seq]


class RNNEncoder(nn.Module):
    def __init__(self, batch_size, rnn_type, nlayers, bidirectional, edim, hdim, zdim, vocab):
        super(RNNEncoder, self).__init__()
        self.w2i = {w: i for (i, w) in enumerate(vocab)}
        self.V = len(self.w2i)

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
        len = output.size(0)
        output = output[-1, :, :]
        return self.encode(output), len



class RNNDecoder(nn.Module):
    def __init__(self, batch_size, rnn_type, nlayers, bidirectional, edim, hdim, zdim, vocab):
        super(RNNDecoder, self).__init__()

        self.w2i = {w: i for (i, w) in enumerate(vocab)}
        self.i2w = {i: w for (w, i) in self.w2i.items()}

        self.V = len(self.w2i)
        self.batch_size = batch_size
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
    def __init__(self, batch_size, rnn_type, nlayers, bidirectional, edim, hdim, zdim, vocab):
        super(VAE, self).__init__()

        self.encoder = RNNEncoder(batch_size, rnn_type, nlayers, bidirectional, edim, hdim, zdim, vocab)
        self.decoder = RNNDecoder(batch_size, rnn_type, nlayers, bidirectional, edim, hdim, zdim, vocab)
        self.project_loc = nn.Linear(zdim, zdim)
        self.project_scale = nn.Linear(zdim, zdim)

        self.nlayers = nlayers
        self.bidirectional = bidirectional
        self.hdim = hdim

    def encode(self, input):
        h0 = torch.Tensor(torch.zeros(self.nlayers + int(self.bidirectional) * self.nlayers,
                                      input.size(1),
                                      self.hdim))
        h, len = self.encoder(input, h0)
        return self.project_loc(h), F.softplus(self.project_scale(h)), len

    def reparametrize(self, loc, scale):
        std = torch.exp(0.5 * scale)
        eps = torch.randn_like(std)
        return loc + eps * std  # z

    def decode(self, input, z):
        return self.decoder(input, z)

    def forward(self, input):
        loc, scale, len = self.encode(input)
        # print('loc, scale', loc.shape, scale.shape, len)
        z = self.reparametrize(loc, scale)
        # print('z', z.shape)
        log_p = self.decode(input, z)
        # print('logp', log_p.shape)
        return log_p.permute(0, 2, 1), loc, scale


def word_prediction_accuracy(logp, target):
    argmax = torch.argmax(logp, dim=1)
    return torch.mean(torch.eq(argmax, target).float())

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(logp, target, loc, scale, anneal_function=None):
    NLL = torch.nn.NLLLoss(ignore_index=0)
    nll_loss = NLL(logp, target)
    kl_loss = -0.5 * torch.sum(1 + scale - loc - scale.exp())
    return nll_loss + kl_loss


def train(model, optimizer, train_split, batch_size, epoch):
    model.train()
    train_loss = 0
    wpa = 0
    n_batches = 0
    for batch_idx, (data, target) in enumerate(train_split.data(batch_size)):
        n_batches += 1
        data = data.to(device)
        target = target.to(device)
        # print(data.shape)
        # print(data)
        optimizer.zero_grad()
        # print(data.shape)
        log_p, loc, scale = model(data)
        # print('loss', log_p.shape, target.shape)

        loss = loss_function(log_p,
                             target,
                             loc,
                             scale)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        wpa += word_prediction_accuracy(log_p, target)
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(train_split), len(train_split),
        #         100. * batch_idx / len(train_split),
        #         loss.item() / len(data)))

    train_loss /= len(train_split)
    wpa /= n_batches
    print('====> Epoch: {} Average training loss: {:.4f}  WPA: {:.4f}'.format(epoch, train_loss, wpa))
    return train_loss, wpa

def validate(model, optimizer, valid_split, batch_size, epoch):
    model.eval()
    valid_loss = 0
    wpa = 0
    n_batches = 0
    for batch_idx, (data, target) in enumerate(valid_split.data(batch_size)):
        n_batches += 1
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        log_p, loc, scale = model(data)
        loss = loss_function(log_p,
                             target,
                             loc,
                             scale)
        loss.backward()
        valid_loss += loss.item()
        optimizer.step()
        wpa += word_prediction_accuracy(log_p, target)

    valid_loss /= len(valid_split)
    wpa /= n_batches
    print('====> Epoch: {} Average validation loss: WPA: {:.4f}'.format(epoch, valid_loss, wpa))
    return valid_loss, wpa


def test(model, test_split, batch_size):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        wpa = 0
        n_batches = 0
        for i, (data, target) in enumerate(test_split.data(batch_size)):
            n_batches += 1
            data = data.to(device)
            target = target.to(device)
            log_p, loc, scale = model(data)
            test_loss += loss_function(log_p, target, loc, scale).item()
            wpa += word_prediction_accuracy(log_p, target)

    test_loss /= len(test_split)
    wpa /= n_batches
    print('====> Test set loss: {:.4f}WPA: {:.4f}'.format(test_loss, wpa))
    return test_loss, wpa

if __name__ == "__main__":

    corpus = Corpus(
        train_path='data/02-21.10way.clean',
        # train_path='data/23.auto.clean',
        validation_path='data/22.auto.clean',
        test_path='data/23.auto.clean'
    )

    batch_size = 64

    print('!!!  V =', len(corpus.vocab))
    model = VAE(batch_size,
                rnn_type='GRU',
                nlayers=1,
                bidirectional=True,
                edim=353,
                hdim=191,
                zdim=13,
                vocab=corpus.vocab)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)


    for epoch in range(1, args.epochs + 1):
        train(model, optimizer, corpus.training, batch_size, epoch)
        validate(model, optimizer, corpus.validation, batch_size, epoch)
        # test(model, corpus.test, batch_size)



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