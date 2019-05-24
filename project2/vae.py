"""
Sentence VAE
------------
Samuel R. Bowman et al., 2015, Generating Sentences from a Continuous Space, SIGNLL

Taking inspiration from https://github.com/pytorch/examples/blob/master/vae/main.py
"""

from __future__ import print_function
import datetime
import os

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchtext.data import BPTTIterator, Field
from torchtext.datasets import PennTreebank
from tqdm import tqdm
from torch.distributions.multivariate_normal import MultivariateNormal
from scipy.special import logsumexp
import numpy as np


EOS = '[EOS]'
BOS = '[BOS]'
UNK = '[UNK]'


def batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def initialize(config):
    print('Corpus initialization...')

    # device = torch.device(config['device'])
    torch.manual_seed(config['seed'])

    field = Field(batch_first=True, tokenize=lambda s: [BOS] + s.split(' ') + [EOS])
    train_corpus = PennTreebank(config['train_path'], field)
    valid_corpus = PennTreebank(config['valid_path'], field)
    test_corpus = PennTreebank(config['test_path'], field)

    field.build_vocab(train_corpus, valid_corpus, test_corpus)

    train_iterator = BPTTIterator(train_corpus,
                            config['batch_size'],
                            config['bptt_len'],
                            device=config['device'],
                            repeat=False)

    valid_iterator = BPTTIterator(valid_corpus,
                                  config['batch_size'],
                                  config['bptt_len'],
                                  device=config['device'],
                                  repeat=False)

    test_iterator = BPTTIterator(test_corpus,
                                  1,
                                  config['bptt_len'],
                                  device=config['device'],
                                  repeat=False)

    vocab = field.vocab

    model = VAE(config['batch_size'],
                rnn_type=config['rnn_type'],
                nlayers=config['num_layers'],
                bidirectional=config['bidir'],
                edim=config['input_emb_dim'],
                hdim=config['hidden_dim'],
                zdim=config['latent_dim'],
                vocab_len=len(vocab),
                word_dropout_prob=config['word_dropout_prob'])

    return model, vocab, train_iterator, valid_iterator, test_iterator


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


class VAE(nn.Module):
    def __init__(self, batch_size, rnn_type, nlayers, bidirectional, edim, hdim, zdim, vocab_len, word_dropout_prob=0.):
        super(VAE, self).__init__()

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
    zdim = loc.size(1)
    loc = loc.squeeze(0)
    var = (scale ** 2).squeeze(0)
    encoder_distribution = MultivariateNormal(loc, torch.diag(var))
    prior_distribution = MultivariateNormal(torch.zeros(zdim), torch.eye(zdim))

    NLL = torch.nn.NLLLoss(ignore_index=0, reduction='sum')
    samples = []
    for s in range(nsamples):
        z = encoder_distribution.sample((1,))                  # sampling a z
        log_q_z_x = encoder_distribution.log_prob(z)           # the probablity of z under the encoder distribution
        log_p_z = prior_distribution.log_prob(z)               # the probability of z under a gaussian prior
        logp = model.decode(sent, z)
        log_p_x_z = - NLL(logp.permute(0, 2, 1), target)  # the sentence probability given the latent variable

        samples.append(log_p_x_z.item() + log_p_z.item() - log_q_z_x.item())
    return np.log(nsamples) - logsumexp(samples)


def train(model, train_data, valid_data):
    annealing = Annealing('linear', 2000)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    if config.get('checkpoint', None) is not None:
        checkpoint = torch.load(config['checkpoint'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        losses = checkpoint['losses']
    else:
        epoch = 0
        losses = []
        wpas = []

    results_dir = config.get('results_dir', str(datetime.datetime.now()).replace(' ', '_')[5:16])
    os.mkdir(os.path.join('pickles', results_dir))
    print('Saving results to:', results_dir)

    print('Starting training!')
    for epoch in tqdm(range(epoch + 1, epoch + config['epochs'] + 1)):
        for i, batch in enumerate(tqdm(train_data)):
            model.train()
            model.encoder.reset_hidden()
            optimizer.zero_grad()

            text, target = batch.text.t(), batch.target.t()
            log_p, loc, scale = model(text)

            loss = loss_function(log_p,
                                 target,
                                 loc,
                                 scale,
                                 annealing)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            wpas.append(word_prediction_accuracy(log_p, target))

            # if i > 0 and i % config['sample_every'] == 0:
                # sample(model, vocab, greedy=True)
                # sample(model, vocab, greedy=False, temp=config['temperature'])
                # perplexity('data/val_lines.txt', model, vocab)


        print('\n====> Epoch: {} Average training loss: {:.4f}  Average WPA: {:.4f}'.format(epoch,
                                                                                          np.mean(losses),
                                                                                          np.mean(wpas)))

        valid_losses, valid_wpas = validate(model, valid_data)
        print('\n====> Epoch: {} Average validation loss: {:.4f}  Average WPA: {:.4f}'.format(epoch,
                                                                                            np.mean(valid_losses),
                                                                                            np.mean(valid_wpas)))

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'losses': losses,
            'valid_losses': valid_losses,
            'wpas': wpas,
            'valid_wpas': valid_wpas
        }, f'pickles/{results_dir}/state_dict_e{epoch}.pt')

    return


def validate(model, valid_data):
    model.eval()
    losses = []
    wpas = []
    with torch.no_grad():
        print('Starting validation!')
        for i, batch in enumerate(tqdm(valid_data)):
            text, target = batch.text.t(), batch.target.t()
            log_p, loc, scale = model(text)
            loss = loss_function(log_p,
                                 target,
                                 loc,
                                 scale,
                                 annealing=None)
            losses.append(loss.item())
            wpas.append(word_prediction_accuracy(log_p, target))
    return losses, wpas


def test(model, test_data, nsamples=16):
    model.eval()
    losses = []
    wpas = []
    model.encoder.reset_hidden(bsz=1)
    with torch.no_grad():
        print('Starting test!')
        for i, batch in enumerate(tqdm(test_data)):
            text, target = batch.text.t(), batch.target.t()
            log_p, loc, scale = model(text)
            losses.append(approximate_sentence_NLL(model, loc, scale, text, target, nsamples))
            wpas.append(word_prediction_accuracy(log_p, target))

    print('\n====> Average test set loss: {:.4f}   Average WPA: {:.4f}'.format(np.mean(losses), np.mean(wpas)))
    return losses, wpas



if __name__ == '__main__':
    config = {
        'train_path': 'data/02-21.10way.clean',
        'valid_path': 'data/22.auto.clean',
        'test_path': 'data/23.auto.clean',
        'batch_size': 64,
        'bptt_len': 40,
        # 'checkpoint': 'pickles/05-23_00:54/state_dict_e1.pt',
        'rnn_type': 'GRU',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'epochs': 1,
        'learning_rate': 1e-3,
        'hidden_dim': 191,
        'input_emb_dim': 353,
        'latent_dim': 13,
        'num_layers': 1,
        'bidir': True,
        'word_dropout_prob': 0.4,
        'importance_samples': 5,
        'seed': 0
    }

    model, vocab, train_iterator, valid_iterator, test_iterator = initialize(config)
    train(model, train_iterator, valid_iterator)
    test(model, test_iterator, config['importance_samples'])
