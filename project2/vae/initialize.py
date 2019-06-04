import nltk
import torch
from torchtext.data import BucketIterator, Field, TabularDataset

from vae.vae import SentenceVAE


def tokenize(s):
    return nltk.Tree.fromstring(s).leaves()


def create_iterator(path, bsz, device, field=None, build_vocab=False):
    if field is None:
        field = Field(batch_first=True,
                      include_lengths=True,
                      init_token='<bos>',
                      eos_token='<eos>',
                      )
    corpus = TabularDataset(path=path,
                            format='tsv',
                            fields=[('text', field)]
                            )
    if build_vocab:
        field.build_vocab(corpus)

    iterator = BucketIterator(dataset=corpus,
                              batch_size=bsz,
                              device=device,
                              repeat=False,
                              shuffle=True,
                              sort=False,
                              sort_key=lambda x: len(x.text),
                              sort_within_batch=True,
                              )

    return iterator, field


def initialize(config):
    print('Corpus initialization...')

    iterator, field = create_iterator(
        config['train_path'], config['batch_size'], config['device'], build_vocab=True
    )

    model = SentenceVAE(config['batch_size'],
                        rnn_type=config['rnn_type'],
                        nlayers=config['num_layers'],
                        bidirectional=config['bidir'],
                        edim=config['input_emb_dim'],
                        hdim=config['hidden_dim'],
                        zdim=config['latent_dim'],
                        vocab_len=len(field.vocab),
                        rnn_dropout=config['rnn_dropout'],
                        word_dropout_prob=config['word_dropout_prob'],
                        device=config['device'])

    if config.get('checkpoint', None) is not None:
        checkpoint = torch.load(config['checkpoint'], map_location=config['device'])
        model.load_state_dict(checkpoint['model_state_dict'])

    return model, iterator, field
