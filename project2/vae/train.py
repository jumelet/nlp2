from __future__ import print_function
import datetime
import os
import torch
import torch.utils.data
from torch import optim
from torchtext.data import BPTTIterator, Field
from torchtext.datasets import PennTreebank
import numpy as np

from vae.vae import SentenceVAE
from vae.metrics import Annealing, approximate_sentence_NLL, elbo_loss, multi_sample_elbo, perplexity, word_prediction_accuracy


EOS = '[EOS]'
BOS = '[BOS]'
UNK = '[UNK]'


def initialize(config):
    print('Corpus initialization...')
    
    device = torch.device(config['device'])
    torch.manual_seed(config['seed'])

    field = Field(batch_first=True, tokenize=lambda s: [BOS] + s.split(' ') + [EOS])
    train_corpus = PennTreebank(config['train_path'], field)
    valid_corpus = PennTreebank(config['valid_path'], field)
    test_corpus = PennTreebank(config['test_path'], field)

    field.build_vocab(train_corpus, valid_corpus, test_corpus)
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


def train(config, model, train_data, valid_data):
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
    print('Starting training!')
    for epoch in range(starting_epoch + 1, starting_epoch + config['epochs'] + 1):
        epoch_losses = []

        ## Batch training
        for i, batch in enumerate(train_data):
            model.train()
            model.encoder.reset_hidden()
            optimizer.zero_grad()

            text, target = batch.text.t(), batch.target.t()
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
        valid_nll, valid_ppl, valid_elbo, valid_wpa = validate(config, model, valid_data)
        print('\n====> Epoch: {} Validation: NLL: {:.4f}  PPL: {:.4f}  ELBO: {:.4f}  WPA: {:.4f}'.format(epoch,
                                                                                                         valid_nll,
                                                                                                         valid_ppl,
                                                                                                         valid_elbo,
                                                                                                         valid_wpa))

        train_losses.append(epoch_train_loss)
        valid_nlls.append(valid_nll)
        valid_ppls.append(valid_ppl)
        valid_elbos.append(valid_elbo)
        valid_wpas.append(valid_wpa)

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

    torch.save({
        'train_losses': train_losses,
        'valid_nlls': valid_nlls,
        'valid_ppls': valid_ppls,
        'valid_elbos': valid_elbos,
        'valid_wpas': valid_wpas
    }, '/home/mariog/projects/nlp2/project2/pickles/{}/statistics.pt'.format(results_dir))

    return


def validate(config, model, valid_data):
    """
    :return: (approximate NLL, validation perplexity, multi-sample elbo, word prediction accuracy)
    """
    model.eval()
    model.encoder.reset_hidden(bsz=1)
    nlls = []
    wpas = []
    elbos = []

    print('Starting validation!')
    for item in valid_data:
        text, target = item.text.t(), item.target.t()
        with torch.no_grad():
            log_p, loc, scale = model(text)
            nll = approximate_sentence_NLL(model, loc, scale, text, target, config['device'], config['importance_samples'])
            wpa = word_prediction_accuracy(model, loc, text, target, config['device'])
            elbo = multi_sample_elbo(loc, scale, nll)
        nlls.append(nll.item())
        wpas.append(wpa.item())
        elbos.append(elbo.item())

    return np.mean(nlls), perplexity(config, nlls), np.mean(elbos), np.mean(wpas)


def test(config, model, test_data):
    nll, ppl, elbo, wpa = validate(config, model, test_data)
    return nll, ppl, elbo, wpa
