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
from torchvision.utils import save_image


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


# todo: replace with DataLoader for sentences
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


class RNNEncoder(nn.Module):
    def __init__(self, rnn_type, nlayers, vocab_size, edim, hdim, zdim, bidirectional=True):
        super(RNNEncoder, self).__init__()
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(vocab_size, hdim, nlayers, bidirectional=bidirectional)
        else:
            raise ValueError("""An invalid option for `--model` was supplied,
                                options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")

        self.embed = nn.Embedding(vocab_size, edim, padding_idx=None)
        self.encode = nn.Linear(hdim, zdim)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        output, hidden =  self.rnn(self.embed(input), hidden)
        return self.encode(output), hidden


class RNNDecoder(nn.Module):
    def __init__(self, rnn_type, nlayers, vocab_size, edim, hdim, zdim, bidirectional=True):
        super(RNNDecoder, self).__init__()
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(vocab_size, hdim, nlayers, bidirectional=bidirectional)
        else:
            raise ValueError("""An invalid option for `--model` was supplied,
                                options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")

        self.embed = nn.Embedding(vocab_size, edim)
        self.decode = nn.Linear(zdim, hdim)
        self.tovocab = nn.Linear(hdim, vocab_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, z):
        output, hidden = self.rnn(self.decode(z), self.embed(z))
        return F.log_softmax(self.tovocab(output)), hidden


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

    def decode(self, input, z):
        return self.decoder(input, z)

    def forward(self, input, hidden):
        loc, scale = self.encode(input, hidden)
        z = self.reparametrize(loc, scale)
        return self.decode(z), loc, scale

#
# Here starts the old, unmodified code (https://github.com/pytorch/examples/blob/master/vae/main.py)
#

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')