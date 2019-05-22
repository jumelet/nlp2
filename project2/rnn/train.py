import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from torchtext.data import BucketIterator, Field, TabularDataset
from tqdm import tqdm

from .rnn import LM


def sample(model, vocab, max_len=20, greedy=True, temp=1.0):
    model.eval()

    hidden = model.init_hidden(1)
    token = torch.Tensor([vocab.stoi['<bos>']]).long()
    sen = []

    for i in range(max_len):
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


def train(config):
    print('Corpus initialization...')
    field = Field(batch_first=True,
                  include_lengths=True,
                  init_token='<bos>',
                  eos_token='<eos>',
                  )
    corpus = TabularDataset(path='project_2_data/train_lines.txt',
                            format='tsv',
                            fields=[('text', field)]
                            )
    field.build_vocab(corpus)

    iterator = BucketIterator(dataset=corpus,
                              batch_size=config['batch_size'],
                              device=config['device'],
                              repeat=False,
                              sort=False,
                              sort_key=lambda x: len(x.text),
                              sort_within_batch=True,
                              )

    vocab_size = len(field.vocab)
    model = LM(
        config['batch_size'],
        vocab_size,
        device=config['device'],
        hidden_size=config['hidden_size'],
        input_emb_dim=config['input_emb_dim'],
        lstm_num_layers=config['num_layers'],
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    print('Starting training!')
    for _ in tqdm(range(config['epochs'])):
        for i, batch in enumerate(tqdm(iterator)):
            model.reset_hidden()

            optimizer.zero_grad()

            text, lengths = batch.text
            lengths -= 1

            input_ = text[:, :-1]
            target = pack_padded_sequence(text[:, 1:], lengths=lengths, batch_first=True)[0]

            logits = model(input_, lengths=lengths)[0].contiguous()
            packed_logits = pack_padded_sequence(logits, lengths=lengths, batch_first=True)[0]

            loss = criterion(packed_logits, target)
            loss.backward()
            optimizer.step()

            if i % config['sample_every'] == 0:
                print('\n', i, loss.item())
                sample(model, field.vocab, greedy=True)
                sample(model, field.vocab, greedy=False, temp=config['temperature'])
