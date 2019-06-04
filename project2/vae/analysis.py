import nltk
import torch
from torch.distributions import MultivariateNormal
from torch.nn import functional as F


def next_token(logp, greedy=True, temp=1.0):
    """
    :param logp: the tensor output of a VAE decoder (last dimension: |V|)
    :param greedy: whether to use greedy decoding (argmax)
    :param temp:
    :return: a tensor containing only the decoded token ids (last dimension: 1)
    """
    if greedy:
        token = torch.argmax(logp, dim=-1)
    else:
        logp = logp ** (1 / temp)
        logp /= logp.sum()
        token = torch.distributions.categorical.Categorical(logp).sample()
    return token


def sample_sentences(config, model, vocab, n=10, max_len=20, greedy=True, temp=1.0):
    """
    Sample sentences from the Sentence VAE by (greedy) decoding from a sample z from the prior distribution.

    :param config: configuration dictionary, typically defined in main.py
    :param model: Sentence VAE
    :param vocab: vocabulary object from torchtext.data.Field
    :param n: number of sentences
    :param max_len: the maximum length of the sampled sentences
    :param greedy: whether to apply greedy decoding
    :param temp: scaling value for non-greedy decoding
    :return: list of n sentences as strings
    """
    device = config['device']
    zdim = config['latent_dim']

    model.eval()
    model.encoder.reset_hidden(bsz=1)
    bos = torch.LongTensor([vocab.stoi['<bos>']]).view(1, 1).to(device)

    prior_distribution = MultivariateNormal(torch.zeros(zdim, device=device), torch.eye(zdim, device=device))

    sentences = []
    for _ in range(n):
        z = prior_distribution.sample((1,))
        tokens = []
        token = bos
        for t in range(max_len):

            tokens.append(vocab.itos[token])
            if vocab.itos[token] == '<eos>':
                break

            with torch.no_grad():
                if t == 0:
                    h0 = model.decoder.decode(z).view(-1, token.size(0), model.decoder.hdim)
                    embed = model.decoder.embed(token)
                    output, h_n = model.decoder.rnn(embed, h0)
                    log_p = F.log_softmax(model.decoder.tovocab(output), dim=-1)
                else:
                    embed = model.decoder.embed(token)
                    output, h_n = model.decoder.rnn(embed, h_n)
                    log_p = F.log_softmax(model.decoder.tovocab(output), dim=-1)

                # print(logp, logp.size())

            token = next_token(log_p, greedy, temp)

        sentences.append(' '.join(tokens))
    return sentences


# embed = model.decoder.embed(input, h)
# model.decoder.rnn()
#
#    if self.rnn_type == 'GRU':
#
#        else:
#            h0 = self.decode(z).view(-1, input.size(0), self.hdim)
#            h0 = (h0, torch.zeros_like(h0, device=self.device))
#        output, _ = self.rnn(self.embed(input), h0)
#
#        log_p = F.log_softmax(self.tovocab(output), dim=-1)


def _reconstruct(z, model, vocab, config, max_len=20):
    """
    Reconstruct a sentence given a latent variable.

    :param input: the input sentence in the form of token ids
    :param z: the latent variable
    :param model: Sentence VAE
    :param vocab: vocabulary object from torchtext.data.Field
    :return: (str) the reconstructed sentence
    """
    greedy = True
    temp = 1.0
    device = config['device']
    zdim = config['latent_dim']

    model.eval()
    model.encoder.reset_hidden(bsz=1)
    bos = torch.LongTensor([vocab.stoi['<bos>']]).view(1, 1).to(device)

    tokens = []
    token = bos
    for t in range(max_len):

        tokens.append(vocab.itos[token])
        if vocab.itos[token] == '<eos>':
            break

        with torch.no_grad():
            if t == 0:
                h0 = model.decoder.decode(z).view(-1, token.size(0), model.decoder.hdim)
                embed = model.decoder.embed(token)
                output, h_n = model.decoder.rnn(embed, h0)
                log_p = F.log_softmax(model.decoder.tovocab(output), dim=-1)
            else:
                embed = model.decoder.embed(token)
                output, h_n = model.decoder.rnn(embed, h_n)
                log_p = F.log_softmax(model.decoder.tovocab(output), dim=-1)

                # print(logp, logp.size())

            token = next_token(log_p, greedy, temp)

    return ' '.join(tokens)


def reconstruct_sentence(sentence, config, model, vocab, nsamples=0):
    """
    Test the reconstruction capability of the Sentence VAE by reconstructing a sentence with samples from
    the approximate posterior or from its mean.

    :param sentence: A PTB-formatted string or a tokenized sentence (w/ whitespaces between tokens)
    :param config: configuration dictionary, typically defined in main.py
    :param model: Sentence VAE
    :param vocab: vocabulary object from torchtext.data.Field
    :param nsamples: number of samples from the approximate posterior. if nsamples = 0, the approximate posterior mean
                     is used instead of sampling
    :return: A list of reconstructed sentences of length nsamples if nsamples > 0, else of length 1.
    """
    device = config['device']
    try:
        tokens = nltk.Tree.fromstring(sentence).leaves()
    except ValueError:
        tokens = sentence.split()
    input_ids = torch.LongTensor([vocab.stoi[w] for w in ['<bos>'] + tokens])
    input_ids = input_ids.view(1, -1).to(config['device'])

    model.eval()
    model.encoder.reset_hidden(bsz=1)

    with torch.no_grad():
        loc, log_var = model.encode(input_ids)
    loc = loc.squeeze(0).to(device)

    reconstructions = []
    if nsamples == 0:
        reconstructions.append(_reconstruct(loc, model, vocab, config))
    else:
        var = torch.exp(log_var).squeeze(0).to(device)
        encoder_distribution = MultivariateNormal(loc, torch.diag(var))
        for _ in range(nsamples):
            z = encoder_distribution.sample((1,))
            reconstructions.append(_reconstruct(z, model, vocab, config))

    return reconstructions


def homotopy(alpha, sentence1, sentence2, vocab, model, config, max_len=20, sample_based=False):
    posteriors = []
    for sentence in [sentence1, sentence2]:
        device = config['device']
        try:
            tokens = nltk.Tree.fromstring(sentence).leaves()
        except ValueError:
            tokens = sentence.split()
        input_ids = torch.LongTensor([vocab.stoi[w] for w in ['<bos>'] + tokens])
        input_ids = input_ids.view(1, -1).to(config['device'])

        model.eval()
        model.encoder.reset_hidden(bsz=1)

        with torch.no_grad():
            loc, log_var = model.encode(input_ids)
        loc = loc.squeeze(0).to(device)

        posteriors.append([loc, log_var])

    posterior_1, posterior_2 = posteriors
    loc1 = posterior_1[0]
    loc2 = posterior_2[0]
    log_var1 = posterior_1[1]
    log_var2 = posterior_2[1]
    if sample_based == True:

        var1 = torch.exp(log_var1).squeeze(0).to(device)
        var2 = torch.exp(log_var2).squeeze(0).to(device)

        encoder_distribution1 = MultivariateNormal(loc1, torch.diag(var1))
        encoder_distribution2 = MultivariateNormal(loc2, torch.diag(var2))
        sample1 = encoder_distribution.sample((1,))
        sample2 = encoder_distribution.sample((1,))
        z_homotopy = alpha * sample1 + (1 - alpha) * sample2
    else:

        z_homotopy = alpha * loc1 + (1 - alpha) * loc1

    reconstruction_homotopy = _reconstruct(z_homotopy, model, vocab, config)

    return reconstruction_homotopy
