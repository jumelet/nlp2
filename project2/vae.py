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
from torchvision import datasets, transforms
from nltk.tree import Tree
from collections import Counter


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


class Split(object):
    def __init__(self, sentences, n=None):
        self.sentences = sentences
        if n is None:
            self.len = len(sentences)
        else:
            self.len = n

    def data(self):
        for s in self.sentences:
            yield s

    def __len__(self):
        return self.len


class CorpusLoader(object):
    def __init__(self, train_path, test_path, validation_path=None):
        counter = Counter([EOS])

        sentences, len, counter = self.load(train_path, counter)
        self.training = Split(sentences, len)

        sentences, len, counter = self.load(train_path, counter)
        self.test = Split(sentences, len)

        if validation_path:
            sentences, len, counter = self.load(validation_path, counter)
            self.validation = Split(sentences, len)

        self.vocab = [w for (w, freq) in counter.most_common()]
        self.w2i = {w: i for (i, w) in enumerate(self.vocab)}
        self.i2w = {i: w for (w, i) in self.w2i.items()}

    def load(self, path, counter=None):
        if counter is None:
            counter = Counter()

        sentences, len = [], 0
        with open(path, 'r') as f:
            for line in f.readlines():
                tree = Tree.fromstring(line)
                sentence = tree.leaves()
                counter += Counter(sentence)
                sentences.append(sentence + [EOS])
                len += 1
        return sentences, len, counter

    def tokens_to_ids(self, seq):
        return [self.w2i[w] for w in seq]

    def ids_to_tokens(self, seq):
        return [self.i2w[i] for i in seq]


class RNNEncoder(nn.Module):
    def __init__(self, rnn_type, nlayers, vocab, edim, hdim, zdim, bidirectional=True):
        super(RNNEncoder, self).__init__()
        self.w2i = {w: i for (i, w) in enumerate(vocab)}
        self.V = len(self.w2i)

        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(self.V, hdim, nlayers, bidirectional=bidirectional)
        else:
            raise ValueError("""An invalid option for `--model` was supplied,
                                options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")

        self.embed = nn.Embedding(self.V, edim, padding_idx=None)
        self.encode = nn.Linear(hdim, zdim)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        output, hidden =  self.rnn(self.embed(input), hidden)
        return self.encode(output), hidden


class RNNDecoder(nn.Module):
    def __init__(self, rnn_type, nlayers, vocab, edim, hdim, zdim, bidirectional=True):
        super(RNNDecoder, self).__init__()
        self.w2i = {w: i for (i, w) in enumerate(vocab)}
        self.i2w = {i: w for (w, i) in self.w2i.items()}
        self.V = len(self.w2i)

        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(self.V, hdim, nlayers, bidirectional=bidirectional)
        else:
            raise ValueError("""An invalid option for `--model` was supplied,
                                options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")


        self.embed = nn.Embedding(self.V, edim)
        self.decode = nn.Linear(zdim, hdim)
        self.tovocab = nn.Linear(hdim, self.V)
        self.eos = self.embed(['[EOS]'])
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, z):
        hidden = self.decode(z)
        seq = [self.w2i['[BOS]']]
        log_p = []
        while seq[-1] != self.w2i['[EOS]']:
            output, hidden = self.rnn(self.eos.view(1, 1, -1), hidden)
            log_p_output = F.log_softmax(self.tovocab(output))
            next_token_id = torch.argmax(log_p_output).item()
            log_p.append(log_p_output)
            seq.append(next_token_id)
        return torch.stack(log_p), torch.stack(seq)


class VAE(nn.Module):
    def __init__(self,
                 rnn_type, nlayers, vocab_size, edim, zdim, hdim, bidirectional=True):
        super(VAE, self).__init__()

        self.encoder = RNNEncoder(rnn_type, nlayers, vocab_size, edim, hdim, zdim, bidirectional)
        self.decoder = RNNDecoder(rnn_type, nlayers, vocab_size, zdim, hdim, bidirectional)
        self.project_loc = nn.Linear(zdim, zdim)
        self.project_scale = nn.Linear(zdim, zdim)

    def encode(self, input, hidden):
        h = self.encoder(input, hidden)
        return self.project_loc(h), self.project_scale(h).softplus()

    def reparametrize(self, loc, scale):
        std = torch.exp(0.5 * scale)
        eps = torch.randn_like(std)
        return loc + eps * std

    def decode(self, z):
        log_p, seq = self.decoder(z)
        return log_p, seq

    def forward(self, input, hidden):
        loc, scale = self.encode(input, hidden)
        z = self.reparametrize(loc, scale)
        log_p, seq = self.decode(z)
        return log_p, seq, loc, scale


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(logp, target, seq, loc, scale, anneal_function=None):
    NLL = torch.nn.NLLLoss()
    nll_loss = NLL(logp, target)
    kl_loss = -0.5 * torch.sum(1 + scale - loc.pow(2) - scale.exp())
    return nll_loss + kl_loss


def train(train_data, valid_data, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_data):
        data = data.to(device)
        optimizer.zero_grad()
        log_p, seq, loc, scale = model(data)
        loss = loss_function(log_p, data, seq, loc, scale).item()
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_data.dataset),
                100. * batch_idx / len(train_data),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_data.dataset)))


def test(test_data, epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_data):
            data = data.to(device)
            log_p, seq, loc, scale = model(data)
            test_loss += loss_function(log_p, data, seq, loc, scale).item()
            # if i == 0:
            #     n = min(data.size(0), 8)
            #     comparison = torch.cat([data[:n],
            #                           recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
            #     save_image(comparison.cpu(),
            #              'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_data)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        # with torch.no_grad():
        #     sample = torch.randn(64, 20).to(device)
        #     sample = model.decode(sample).cpu()
        #     save_image(sample.view(64, 1, 28, 28),
        #                'results/sample_' + str(epoch) + '.png')