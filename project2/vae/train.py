from __future__ import print_function
import datetime
import os

import nltk
import torch
import torch.utils.data
from torch import optim
from torchtext.data import BPTTIterator, Field
from torchtext.datasets import PennTreebank
import numpy as np

from vae.vae import SentenceVAE
from vae.metrics import Annealing, approximate_sentence_NLL, elbo_loss, multi_sample_elbo, perplexity_, word_prediction_accuracy


EOS = '<eos>'
BOS = '<bos>'


def tokenize(s):
    return nltk.Tree.fromstring(s).leaves()


def initialize(config):
    print('Corpus initialization...')
    
    device = torch.device(config['device'])
    torch.manual_seed(config['seed'])

    field = Field(batch_first=True, tokenize=tokenize, init_token=BOS, eos_token=EOS)
    train_corpus = PennTreebank(config['train_path'], field)
    valid_corpus = PennTreebank(config['valid_path'], field)
    test_corpus = PennTreebank(config['test_path'], field)

    field.build_vocab(train_corpus, valid_corpus, test_corpus, min_freq=1)
    vocab = field.vocab

    train_iterator = BPTTIterator(train_corpus,
                            config['batch_size'],
                            config['bptt_len'],
                            device=config['device'],
                            repeat=False)

    valid_iterator = BPTTIterator(valid_corpus,
                                  1,
                                  config['bptt_len'],
                                  device=config['device'],
                                  repeat=False)

    test_iterator = BPTTIterator(test_corpus,
                                  1,
                                  config['bptt_len'],
                                  device=config['device'],
                                  repeat=False)

    model = SentenceVAE(config['batch_size'],
                rnn_type=config['rnn_type'],
                nlayers=config['num_layers'],
                bidirectional=config['bidir'],
                edim=config['input_emb_dim'],
                hdim=config['hidden_dim'],
                zdim=config['latent_dim'],
                vocab_len=len(vocab),
                word_dropout_prob=config['word_dropout_prob'],
                device=device)

    return model, vocab, train_iterator, valid_iterator, test_iterator


def train(config, model, train_data, valid_data, vocab):
    annealing = Annealing('linear', 2000)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    if config.get('checkpoint', None) is not None:
        checkpoint = torch.load(config['checkpoint'])
        model.load_state_dict(checkpoint['model_state_dict'])
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
    os.mkdir(os.path.join('pickles', results_dir))
    print('Saving results to:', results_dir)

    best_epoch = (float('inf'), 0)
    bos_for_batch = torch.LongTensor([vocab.stoi[BOS]]).repeat(config['batch_size'], 1).to(config['device'])
    eos_for_batch = torch.LongTensor([vocab.stoi[EOS]]).repeat(config['batch_size'], 1).to(config['device'])

    print('Starting training!')
    for epoch in range(starting_epoch + 1, starting_epoch + config['epochs'] + 1):
        epoch_losses = []

        ## Training (with batches)
        for i, batch in enumerate(train_data):
            model.train()
            model.encoder.reset_hidden()
            optimizer.zero_grad()

            tokens = batch.text.t()
            text = torch.cat((bos_for_batch, tokens), dim=1)
            target = torch.cat((tokens, eos_for_batch), dim=1)

            log_p, loc, scale = model(text)

            loss = elbo_loss(log_p,
                             target,
                             loc,
                             scale,
                             annealing)
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


def validate(config, model, valid_data, vocab, phase='validation', verbose=True):
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
    print('Starting {}!'.format(phase))
    for item in valid_data:
        tokens = item.text.t()
        text = torch.cat((bos_for_item, tokens), dim=1)
        target = torch.cat((tokens, eos_for_item), dim=1)
        with torch.no_grad():
            log_p, loc, scale = model(text)
            nll = approximate_sentence_NLL(model, loc, scale, text, target, config['device'], config['importance_samples'])
            wpa = word_prediction_accuracy(model, loc, text, target, config['device'])
            elbo = multi_sample_elbo(loc, scale, nll)
        nlls.append(nll.item())
        wpas.append(wpa.item())
        elbos.append(elbo.item())

    ppl_path = config['valid_path'] if phase == 'validation' else config['test_path']
    avg_ppl = perplexity_(config, model, ppl_path, vocab)

    if verbose:
        print('\n====> {}: NLL: {:.4f}  PPL: {:.4f}  ELBO: {:.4f}  WPA: {:.4f}'.format(
            phase, np.mean(nlls), avg_ppl, np.mean(elbos), np.mean(wpas))
        )
    return np.mean(nlls), avg_ppl, np.mean(elbos), np.mean(wpas)


def test(config, model, test_data, vocab):
    nll, ppl, elbo, wpa = validate(config, model, test_data, vocab, phase='test')
    return nll, ppl, elbo, wpa
