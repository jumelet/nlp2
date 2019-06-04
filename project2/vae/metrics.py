from __future__ import print_function
import numpy as np
import torch
import nltk
from torch.distributions.multivariate_normal import MultivariateNormal
from scipy.special import logsumexp


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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
            return sigmoid((self.step - 2500) / 400)

    def reset(self):
        self.step = 0


def KLLoss(loc, scale):
    kl_loss = -0.5 * torch.sum(1 + (scale ** 2) - (loc ** 2) - (scale ** 2).exp())
    return kl_loss


def elbo_loss(logp, target, loc, scale):
    NLL = torch.nn.NLLLoss(ignore_index=0, reduction='sum')
    nll_loss = NLL(logp, target)
    kl_loss = KLLoss(loc, scale)

    return nll_loss, kl_loss


def word_prediction_accuracy(model, loc, input, target, device):
    """
    Word prediction accuracy, i.e. the fraction of correctly predicted words, using greedy decoding.
    We follow the common heuristic of using the mean of the approximate posterior instead of a sample.
    """
    loc = loc.squeeze(0).to(device)
    logp = model.decode(input, loc)
    return torch.mean(torch.eq(torch.argmax(logp, dim=-1), target).float())


def multi_sample_elbo(loc, scale, approximate_nll):
    """
    :param loc: The location vector of this sentence's latent variable
    :param scale: The scale vector of this sentence's latent variable
    :param approximate_nll: the importance sampling estimate of this sentence's NLL

    :return: A multi-sample estimate of the evidence lower-bound (ELBO) for the sentence VAE.
    """
    return approximate_nll + KLLoss(loc, scale)


def approximate_sentence_NLL(model, loc, scale, sent, target, device, nsamples=16):
    """
    NLL with Importance Sampling.
    """
    zdim = loc.size(1)
    loc = loc.squeeze(0).to(device)
    var = (scale ** 2).squeeze(0).to(device)
    encoder_distribution = MultivariateNormal(loc, torch.diag(var))
    prior_distribution = MultivariateNormal(torch.zeros(zdim, device=device), torch.eye(zdim, device=device))

    NLL = torch.nn.NLLLoss(reduction='sum')
    samples = []
    for s in range(nsamples):
        z = encoder_distribution.sample((1,))             # sampling a z
        log_q_z_x = encoder_distribution.log_prob(z)      # the probablity of z under the encoder distribution
        log_p_z = prior_distribution.log_prob(z)          # the probability of z under a gaussian prior
        with torch.no_grad():
            logp = model.decode(sent, z)[0]               # the log-softmax word probabilities
        log_p_x_z = - NLL(logp, target)  # the sentence probability given the latent variable
        samples.append(log_p_x_z.item() + log_p_z.item() - log_q_z_x.item())
    return np.log(nsamples) - logsumexp(samples)
