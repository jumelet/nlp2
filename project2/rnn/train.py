import nltk
import torch
import torch.optim as optim
from torchtext.data import BPTTIterator, Field
from torchtext.datasets import PennTreebank
from tqdm import tqdm

from .rnn import LM


def tokenize(s):
    return nltk.Tree.fromstring(s).leaves()


def sample(model, max_len=20, greedy=20):
    hidden = model.init_hidden(1)
    # tbc


def train(config):
    print('Corpus initialization...')
    field = Field(use_vocab=True, tokenize=tokenize)
    corpus = PennTreebank('project_2_data/02-21.10way.clean', field)
    field.build_vocab(corpus)

    iterator = BPTTIterator(corpus,
                            batch_size=config['batch_size'],
                            bptt_len=config['bptt_len'],
                            device=config['device'],
                            repeat=False,
                            )

    vocab_size = len(field.vocab)
    model = LM(
        config['batch_size'],
        vocab_size,
        device=config['device']
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    print('Starting training!')
    for batch in tqdm(iterator):
        optimizer.zero_grad()

        text, targets = batch.text.t(), batch.target.t().contiguous()

        logits = model(text).contiguous()

        loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()

        print(loss.item())
