import torch
from torch.distributions import MultivariateNormal


def sample_sentences(config, model, vocab, n=10, max_len=20, greedy=True, temp=1.0):
    """
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

            if greedy:
                token = torch.argmax(logp, dim=-1)
            else:
                logp = logp ** (1 / temp)
                logp /= logp.sum()
                token = torch.distributions.categorical.Categorical(logp).sample()

        sentences.append(' '.join(tokens))
    return sentences
