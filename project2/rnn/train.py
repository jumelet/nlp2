import datetime
import os

import torch
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from torchtext.data import BPTTIterator, BucketIterator, Field, TabularDataset
from torchtext.datasets import PennTreebank
from tqdm import tqdm

from .rnn import LM
from .sample import perplexity, sample


def initialize(config):
    print('Corpus initialization...')
    if config.get('bptt_len', None) is not None:
        field = Field(batch_first=True,
                      tokenize=lambda s: ['<bos>'] + s.split(' ')
                      )
        corpus = PennTreebank('data/train_lines.txt', field)
    else:
        field = Field(batch_first=True,
                      include_lengths=True,
                      init_token='<bos>',
                      eos_token='<eos>',
                      )
        corpus = TabularDataset(path='data/train_lines.txt',
                                format='tsv',
                                fields=[('text', field)]
                                )
    field.build_vocab(corpus)

    if config.get('bptt_len', None) is not None:
        iterator = BPTTIterator(corpus,
                                config['batch_size'],
                                config['bptt_len'],
                                device=config['device'],
                                repeat=False,
                                )
    else:
        iterator = BucketIterator(dataset=corpus,
                                  batch_size=config['batch_size'],
                                  device=config['device'],
                                  repeat=False,
                                  sort=False,
                                  sort_key=lambda x: len(x.text),
                                  sort_within_batch=True,
                                  )

    vocab = field.vocab
    model = LM(
        config['batch_size'],
        len(vocab),
        device=config['device'],
        hidden_size=config['hidden_size'],
        input_emb_dim=config['input_emb_dim'],
        lstm_num_layers=config['num_layers'],
    )

    return vocab, iterator, model


def train(config):
    vocab, iterator, model = initialize(config)

    model.train()
    criterion = torch.nn.CrossEntropyLoss()
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

    results_dir = config.get('results_dir', str(datetime.datetime.now()).replace(' ', '_')[5:16])
    os.mkdir(os.path.join('pickles', results_dir))
    print('Saving results to:', results_dir)

    print('Starting training!')
    for epoch in tqdm(range(epoch+1, epoch+config['epochs']+1)):
        for i, batch in enumerate(tqdm(iterator)):
            model.reset_hidden()

            optimizer.zero_grad()

            if config.get('bptt_len', None) is None:
                text, lengths = batch.text
                lengths -= 1

                input_ = text[:, :-1]
                target = pack_padded_sequence(text[:, 1:], lengths=lengths, batch_first=True)[0]

                logits = model(input_, lengths=lengths)[0]
                logits = pack_padded_sequence(logits, lengths=lengths, batch_first=True)[0]
            else:
                text, target = batch.text.t(), batch.target.view(-1)
                logits = model(text)[0].view(-1, len(vocab))

            loss = criterion(logits, target)
            loss.backward()

            optimizer.step()

            losses.append(loss.item())
            if i > 0 and i % config['sample_every'] == 0:
                print('\n', i, loss.item())
                sample(model, vocab, greedy=True)
                sample(model, vocab, greedy=False, temp=config['temperature'])
                perplexity('data/val_lines.txt', model, vocab)

            break

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'losses': losses,
        }, f'pickles/{results_dir}/state_dict_e{epoch}.pt')
