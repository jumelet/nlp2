from __future__ import print_function

import datetime
import os

import numpy as np
import torch
import torch.utils.data
from torch import optim
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm

from vae.metrics import (
    Annealing, approximate_sentence_NLL, elbo_loss, multi_sample_elbo, word_prediction_accuracy)


EOS = '<eos>'
BOS = '<bos>'


def train(config, model, train_data, val_iterator):
    annealing = Annealing(config['annealing'])
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    if config.get('checkpoint', None) is not None:
        checkpoint = torch.load(config['checkpoint'], map_location=config['device'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch']
    else:
        starting_epoch = 0

    train_losses = []
    train_elbos = []
    train_kls = []
    train_nlls = []
    valid_nlls = []
    valid_ppls = []
    valid_elbos = []
    valid_wpas = []

    results_dir = config.get('results_dir', str(datetime.datetime.now()).replace(' ', '_')[5:16])
    if not os.path.exists(os.path.join('pickles', results_dir)):
        os.mkdir(os.path.join('pickles', results_dir))
    print('Saving results to:', results_dir)
    if config.get('use_tb', False):
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(os.path.join(config['logdir'], results_dir, 'SenVAE'))

    best_epoch = (float('inf'), 0)

    print('Starting training!')
    for epoch in range(starting_epoch + 1, starting_epoch + config['epochs'] + 1):
        epoch_losses = []

        ## Training (with batches)
        for i, batch in enumerate(tqdm(train_data)):
            text, lengths = batch.text
            bsz = text.shape[0]

            lengths -= 1

            input_ = text[:, :-1]

            target = pack_padded_sequence(text[:, 1:], lengths=lengths, batch_first=True)[0]

            model.train()
            model.encoder.reset_hidden(bsz)
            optimizer.zero_grad()

            log_p, loc, scale = model(input_)
            log_p = pack_padded_sequence(log_p, lengths=lengths, batch_first=True)[0]

            nll, kl = elbo_loss(log_p,
                                target,
                                loc,
                                scale)

            loss = nll + kl*annealing.rate()

            train_elbos.append(loss.item()/bsz)
            train_kls.append(kl.item()/bsz)
            train_nlls.append(nll.item()/bsz)

            if config.get('use_tb', False):
                writer.add_scalar('NLL', nll.item()/bsz, i)
                writer.add_scalar('KL', kl.item()/bsz, i)
                writer.add_scalar('ELBO', loss/bsz, i)

            loss.backward()
            epoch_losses.append(loss.item()/bsz)
            optimizer.step()

            break

        epoch_train_loss = np.mean(epoch_losses)
        print('\n====> Epoch: {} Average training loss: {:.4f}'.format(epoch, epoch_train_loss))

        ## Validation
        valid_nll, valid_ppl, valid_elbo, valid_wpa = validate(config, model, val_iterator)

        ## Collect training statistics
        train_losses.append(epoch_train_loss)
        valid_nlls.append(valid_nll)
        valid_ppls.append(valid_ppl)
        valid_elbos.append(valid_elbo)
        valid_wpas.append(valid_wpa)

        # Store model if it's the best so far
        if valid_ppl <= best_epoch[0]:
            pickles_path = 'pickles'
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
        # torch.save({
        #     'train_losses': train_losses,
        #     'train_elbos': train_elbos,
        #     'train_kls': train_kls,
        #     'train_nlls': train_nlls,
        #     'valid_nlls': valid_nlls,
        #     'valid_ppls': valid_ppls,
        #     'valid_elbos': valid_elbos,
        #     'valid_wpas': valid_wpas
        # }, 'pickles/{}/statistics.pt'.format(results_dir))

    return


def validate(config, model, iterator, phase='val', verbose=True):
    """
    :return: (approximate NLL, validation perplexity, multi-sample elbo, word prediction accuracy)
    """
    model.eval()

    nlls = []
    wpas = []
    elbos = []
    ntokens = 0

    for batch in tqdm(iterator):
        text, lengths = batch.text
        bsz = text.shape[0]

        model.encoder.reset_hidden(bsz)

        lengths -= 1

        input_ = text[:, :-1]

        target = pack_padded_sequence(text[:, 1:], lengths=lengths, batch_first=True)[0]

        NLL = torch.nn.NLLLoss(reduction='sum')

        with torch.no_grad():
            z, loc, scale = model(input_)

            nll0 = NLL(z[0], target)

            nll = approximate_sentence_NLL(model, loc, scale, input_, target, config['device'],
                                           config['importance_samples'])
            nll2 = approximate_sentence_NLL(model, loc, scale, input_, target, config['device'],
                                            100)
            wpa = word_prediction_accuracy(model, loc, input_, target, config['device'])
            elbo = multi_sample_elbo(loc, scale, nll)
        nlls.append(nll.item())
        wpas.append(wpa.item())
        elbos.append(elbo.item())
        ntokens += input_.shape[1]

    ppl = np.exp(np.sum(nlls) / ntokens)

    if verbose:
        print('\n====> {}: NLL: {:.4f}  PPL: {:.4f}  ELBO: {:.4f}  WPA: {:.4f}'.format(
            phase, np.mean(nlls), ppl, np.mean(elbos), np.mean(wpas))
        )
    return np.mean(nlls), ppl, np.mean(elbos), np.mean(wpas)
