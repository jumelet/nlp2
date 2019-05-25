from __future__ import print_function
import datetime
import os
import torch
import torch.utils.data
from torch import optim
from torch.distributions.multivariate_normal import MultivariateNormal
from torchtext.data import BPTTIterator, Field
from torchtext.datasets import PennTreebank
from scipy.special import logsumexp
import numpy as np
from tqdm import tqdm

from vae.vae import SentenceVAE


EOS = '[EOS]'
BOS = '[BOS]'
UNK = '[UNK]'


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


def approximate_sentence_NLL(model, loc, scale, sent, target, device, nsamples=16):
    """
        NLL with Importance Sampling
    """
    zdim = loc.size(1)
    loc = loc.squeeze(0).to(device)
    var = (scale ** 2).squeeze(0).to(device)
    encoder_distribution = MultivariateNormal(loc, torch.diag(var))
    prior_distribution = MultivariateNormal(torch.zeros(zdim).to(device), torch.eye(zdim).to(device))

    NLL = torch.nn.NLLLoss(ignore_index=0, reduction='sum')
    samples = []
    for s in range(nsamples):
        z = encoder_distribution.sample((1,))              # sampling a z
        log_q_z_x = encoder_distribution.log_prob(z)           # the probablity of z under the encoder distribution
        log_p_z = prior_distribution.log_prob(z)               # the probability of z under a gaussian prior
        logp = model.decode(sent, z)
        log_p_x_z = - NLL(logp.permute(0, 2, 1), target)  # the sentence probability given the latent variable

        samples.append(log_p_x_z.item() + log_p_z.item() - log_q_z_x.item())
    return np.log(nsamples) - logsumexp(samples)


def initialize(config):
    print('Corpus initialization...')
    
    device = torch.device(config['device'])
    torch.manual_seed(config['seed'])

    field = Field(batch_first=True, tokenize=lambda s: [BOS] + s.split(' ') + [EOS])
    train_corpus = PennTreebank(config['train_path'], field)
    valid_corpus = PennTreebank(config['valid_path'], field)
    test_corpus = PennTreebank(config['test_path'], field)

    field.build_vocab(train_corpus, valid_corpus, test_corpus)
    vocab = field.vocab

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

    model = SentenceVAE(config['batch_size'],
                rnn_type=config['rnn_type'],
                nlayers=config['num_layers'],
                bidirectional=config['bidir'],
                edim=config['input_emb_dim'],
                hdim=config['hidden_dim'],
                zdim=config['latent_dim'],
                vocab_len=len(vocab),
                word_dropout_prob=config['word_dropout_prob'],
                device=device)

    return model, vocab, train_iterator, valid_iterator, test_iterator


def train(config, model, train_data, valid_data):
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
        avg_train_losses = []
        avg_train_wpas = []
        avg_valid_losses = []
        avg_valid_wpas = []

    results_dir = config.get('results_dir', str(datetime.datetime.now()).replace(' ', '_')[5:16])
    os.mkdir(os.path.join('pickles', results_dir))
    print('Saving results to:', results_dir)

    lowest_epoch_loss = (float('inf'), 0)
    print('Starting training!')
    for epoch in range(epoch + 1, epoch + config['epochs'] + 1):
        epoch_losses = []
        epoch_wpas = []
        for i, batch in enumerate(train_data):
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

            epoch_losses.append(loss.item())
            epoch_wpas.append(word_prediction_accuracy(log_p, target).item())

        avg_epoch_loss, avg_epoch_wpa = np.mean(epoch_losses), np.mean(epoch_wpas)
        print('\n====> Epoch: {} Average training loss: {:.4f}  Average WPA: {:.4f}'.format(epoch,
                                                                                            avg_epoch_loss,
                                                                                            avg_epoch_wpa))
        valid_epoch_losses, valid_epoch_wpas = validate(model, valid_data)
        avg_valid_epoch_loss, avg_valid_epoch_wpa = np.mean(valid_epoch_losses), np.mean(valid_epoch_wpas)
        print('\n====> Epoch: {} Average validation loss: {:.4f}  Average WPA: {:.4f}'.format(epoch,
                                                                                              avg_valid_epoch_loss,
                                                                                              avg_valid_epoch_wpa))

        avg_train_losses.append(avg_epoch_loss)
        avg_train_wpas.append(avg_epoch_wpa)
        avg_valid_losses.append(avg_valid_epoch_loss)
        avg_valid_wpas.append(avg_valid_epoch_wpa)

        if avg_valid_epoch_loss <= lowest_epoch_loss[0]:
            print('!!!')
            pickles_path = '/home/mariog/projects/nlp2/project2/pickles'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, '{}/{}/state_dict_e{}.pt'.format(pickles_path, results_dir, epoch))

            lowest_epoch_loss = (avg_valid_epoch_loss, epoch)
            old_path = '{}/{}/state_dict_e{}.pt'.format(pickles_path, results_dir, lowest_epoch_loss[1])
            if os.path.exists(old_path):
                os.remove(old_path)

    torch.save({
        'avg_train_losses': avg_train_losses,
        'avg_train_wpas': avg_train_wpas,
        'avg_valid_losses': avg_valid_losses,
        'avg_valid_wpas': avg_valid_wpas
    }, '/home/mariog/projects/nlp2/project2/pickles/{}/statistics.pt'.format(results_dir))

    return


def validate(model, valid_data):
    model.eval()
    losses = []
    wpas = []

    print('Starting validation!')
    with torch.no_grad():
        for i, batch in enumerate(valid_data):
            text, target = batch.text.t(), batch.target.t()
            log_p, loc, scale = model(text)
            loss = loss_function(log_p,
                                 target,
                                 loc,
                                 scale,
                                 annealing=None)
            losses.append(loss.item())
            wpas.append(word_prediction_accuracy(log_p, target).item())
    return losses, wpas


def test(config, model, test_data):
    try:
        nsamples = config['importance_samples']
    except KeyError:
        nsamples = 16

    model.eval()
    model.encoder.reset_hidden(bsz=1)
    losses = []
    wpas = []

    print('Starting test!')
    with torch.no_grad():
        for i, batch in enumerate(test_data):
            text, target = batch.text.t(), batch.target.t()
            log_p, loc, scale = model(text)
            losses.append(approximate_sentence_NLL(model, loc, scale, text, target, config['device'], nsamples).item())
            wpas.append(word_prediction_accuracy(log_p, target).item())

    print('\n====> Average test set loss: {:.4f}   Average WPA: {:.4f}'.format(np.mean(losses), np.mean(wpas)))
    return losses, wpas


