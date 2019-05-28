import torch
from torch.distributions import MultivariateNormal


def sample_sentences(config, model, vocab, max_len=20, n=10, greedy=True, temp=1.0):
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
