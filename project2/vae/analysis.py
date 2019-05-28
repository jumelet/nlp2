import nltk
import torch
from torch.distributions import MultivariateNormal


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
    z = prior_distribution.sample((1,))

    sentences = []
    for _ in range(n):
        tokens = []
        token = bos
        for _ in range(max_len):
            tokens.append(vocab.itos[token])
            if vocab.itos[token] == '<eos>':
                break

            with torch.no_grad():
                logp = model.decode(token, z)

            token = next_token(logp, greedy, temp)

        sentences.append(' '.join(tokens))
    return sentences


def _reconstruct(input, z, model, vocab):
    """
    Reconstruct a sentence given a latent variable.

    :param input: the input sentence in the form of token ids
    :param z: the latent variable
    :param model: Sentence VAE
    :param vocab: vocabulary object from torchtext.data.Field
    :return: (str) the reconstructed sentence
    """
    with torch.no_grad():
        logp = model.decode(input, z)
    token_ids = torch.argmax(logp, dim=-1)
    return ' '.join([vocab.itos[i] for i in token_ids.squeeze(0)])


def reconstruct_sentence(sentence, config, model, vocab, nsamples=10):
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
        loc, scale = model.encode(input_ids)
    loc = loc.squeeze(0).to(device)

    reconstructions = []
    if nsamples == 0:
        reconstructions.append(_reconstruct(input_ids, loc, model, vocab))
    else:
        var = (scale ** 2).squeeze(0).to(device)
        encoder_distribution = MultivariateNormal(loc, torch.diag(var))
        for _ in range(nsamples):
            z = encoder_distribution.sample((1,))
            reconstructions.append(_reconstruct(input_ids, z, model, vocab))

    return reconstructions
