from __future__ import print_function

import datetime
import os

import nltk
import numpy as np
import torch
import torch.utils.data
from tensorboardX import SummaryWriter
from torch import optim
from torch.nn.utils.rnn import pack_padded_sequence
from torchtext.data import BucketIterator, Field, TabularDataset
from tqdm import tqdm

from vae.metrics import (
    Annealing, approximate_sentence_NLL, elbo_loss, multi_sample_elbo, perplexity,
    word_prediction_accuracy)
from vae.vae import SentenceVAE

EOS = '<eos>'
BOS = '<bos>'


def tokenize(s):
    return nltk.Tree.fromstring(s).leaves()


def initialize(config):
    print('Corpus initialization...')
    field = Field(batch_first=True,
                  include_lengths=True,
                  init_token='<bos>',
                  eos_token='<eos>',
                  )
    corpus = TabularDataset(path=config['train_path'],
                            format='tsv',
                            fields=[('text', field)]
                            )
    field.build_vocab(corpus)

    iterator = BucketIterator(dataset=corpus,
                              batch_size=config['batch_size'],
                              device=config['device'],
                              repeat=False,
                              shuffle=True,
                              sort=False,
                              sort_key=lambda x: len(x.text),
                              sort_within_batch=True,
                              )

    vocab = field.vocab

    model = SentenceVAE(config['batch_size'],
                        rnn_type=config['rnn_type'],
                        nlayers=config['num_layers'],
                        bidirectional=config['bidir'],
                        edim=config['input_emb_dim'],
                        hdim=config['hidden_dim'],
                        zdim=config['latent_dim'],
                        vocab_len=len(vocab),
                        rnn_dropout=config['rnn_dropout'],
                        word_dropout_prob=config['word_dropout_prob'],
                        device=config['device'])

    return model, vocab, iterator


def train(config, model, train_data, valid_data, vocab):
    annealing = Annealing(config['annealing'])
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    if config.get('checkpoint', None) is not None:
        checkpoint = torch.load(config['checkpoint'], map_location=config['device'])
        model.load_state_dict(checkpoint['model_state_dict'], )
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch']
    else:
        starting_epoch = 0

    train_losses = []
    valid_nlls = []
    valid_ppls = []
    valid_elbos = []
    valid_wpas = []

    results_dir = config.get('results_dir', str(datetime.datetime.now()).replace(' ', '_')[5:16])
    # os.mkdir(os.path.join('pickles', results_dir))
    # print('Saving results to:', results_dir)
    writer = SummaryWriter(os.path.join(config['logdir'], results_dir, 'SenVAE'))

    best_epoch = (float('inf'), 0)

    print('Starting training!')
    for epoch in range(starting_epoch + 1, starting_epoch + config['epochs'] + 1):
        epoch_losses = []

        ## Training (with batches)
        for i, batch in tqdm(enumerate(train_data)):
            model.train()
            model.encoder.reset_hidden()
            optimizer.zero_grad()

            text, lengths = batch.text
            bsz = text.shape[0]
            lengths -= 1

            input_ = text[:, :-1]

            target = pack_padded_sequence(text[:, 1:], lengths=lengths, batch_first=True)[0]

            log_p, loc, scale = model(input_)
            log_p = pack_padded_sequence(log_p, lengths=lengths, batch_first=True)[0]

            nll, kl = elbo_loss(log_p,
                                target,
                                loc,
                                scale)

            writer.add_scalar('NLL', nll.item()/bsz, i)
            writer.add_scalar('KL', kl.item()/bsz, i)

            loss = nll + kl*annealing.rate()

            writer.add_scalar('ELBO', loss/bsz, i)

            loss.backward()
            epoch_losses.append(loss.item())
            optimizer.step()

        epoch_train_loss = np.mean(epoch_losses)
        print('\n====> Epoch: {} Average training loss: {:.4f}'.format(epoch, epoch_train_loss))

        ## Validation
        valid_nll, valid_ppl, valid_elbo, valid_wpa = validate(config, model, valid_data, vocab)

        ## Collect training statistics
        train_losses.append(epoch_train_loss)
        valid_nlls.append(valid_nll)
        valid_ppls.append(valid_ppl)
        valid_elbos.append(valid_elbo)
        valid_wpas.append(valid_wpa)

        ## Store model if it's the best so far
        if valid_ppl <= best_epoch[0]:
            pickles_path = '/home/mariog/projects/nlp2/project2/pickles'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, '{}/{}/state_dict_e{}.pt'.format(pickles_path, results_dir, epoch))

            old_path = '{}/{}/state_dict_e{}.pt'.format(pickles_path, results_dir, best_epoch[1])
            if os.path.exists(old_path):
                os.remove(old_path)
            best_epoch = (valid_ppl, epoch)

    ## Finally store all learning statistics
    torch.save({
        'train_losses': train_losses,
        'valid_nlls': valid_nlls,
        'valid_ppls': valid_ppls,
        'valid_elbos': valid_elbos,
        'valid_wpas': valid_wpas
    }, '/home/mariog/projects/nlp2/project2/pickles/{}/statistics.pt'.format(results_dir))

    return


def validate(config, model, iterator, vocab, phase='validation', verbose=True):
    """
    :return: (approximate NLL, validation perplexity, multi-sample elbo, word prediction accuracy)
    """
    model.eval()
    model.encoder.reset_hidden(bsz=1)
    bos_for_item = torch.LongTensor([vocab.stoi[BOS]]).view(1, 1).to(config['device'])
    eos_for_item = torch.LongTensor([vocab.stoi[EOS]]).view(1, 1).to(config['device'])

    nlls = []
    wpas = []
    elbos = []
    for item in iterator:
        tokens = item.text.t()
        text = torch.cat((bos_for_item, tokens), dim=1)
        target = torch.cat((tokens, eos_for_item), dim=1)
        with torch.no_grad():
            log_p, loc, scale = model(text)
            nll = approximate_sentence_NLL(model, loc, scale, text, target, config['device'],
                                           config['importance_samples'])
            wpa = word_prediction_accuracy(model, loc, text, target, config['device'])
            elbo = multi_sample_elbo(loc, scale, nll)
        nlls.append(nll.item())
        wpas.append(wpa.item())
        elbos.append(elbo.item())

    ppl = perplexity(config, model, vocab, phase)

    if verbose:
        print('\n====> {}: NLL: {:.4f}  PPL: {:.4f}  ELBO: {:.4f}  WPA: {:.4f}'.format(
            phase, np.mean(nlls), ppl, np.mean(elbos), np.mean(wpas))
        )
    return np.mean(nlls), ppl, np.mean(elbos), np.mean(wpas)


def test(config, model, test_data, vocab):
    nll, ppl, elbo, wpa = validate(config, model, test_data, vocab, phase='test')
    return nll, ppl, elbo, wpa
