import torch.nn as nn
import torch


def sample(model, vocab, max_len=20, greedy=True, temp=1.0):
    model.eval()

    hidden = model.init_hidden(1)
    token = torch.Tensor([vocab.stoi['<bos>']]).long()
    sen = []

    for i in range(max_len):
        with torch.no_grad():
            out, hidden = model(token.view(1, -1), hidden=hidden)
        probs = nn.functional.softmax(out[0])

        if greedy:
            token = torch.argmax(probs)
        else:
            probs = probs ** (1 / temp)
            probs /= probs.sum()
            token = torch.distributions.categorical.Categorical(probs).sample()
        sen.append(vocab.itos[token])
        if vocab.itos[token] == '<eos>':
            break

    print(' '.join(sen))

    model.train()


def perplexity(path, model, vocab, device='cpu'):
    with open(path) as f:
        lines = [['<bos>'] + l.strip().split(' ') + ['<eos>'] for l in f.read().strip().split('\n')]

    model.eval()

    probs = 0.
    n = 0

    CEL = nn.CrossEntropyLoss(reduction='none')

    for line in lines:
        sen = torch.LongTensor([vocab.stoi[w] for w in line])
        with torch.no_grad():
            out, hidden = model(sen[:-1].view(1, -1),
                                hidden=model.init_hidden(1))

        probs += CEL(out[0], sen[1:].reshape(-1).to(device))

        n += len(line[:-1])

    avg_perp = torch.exp(probs / n)

    print(avg_perp)

    model.train()

    return avg_perp
