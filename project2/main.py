import argparse
import json
from vae.train import train, validate
from vae.analysis import sample_sentences, reconstruct_sentence
from vae.initialize import initialize, create_iterator
from pprint import pprint

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Sentence VAE')
    parser.add_argument('-c', '--config', help='Config dictionary in a txt file', required=True)
    config_path = parser.parse_args().config

    with open(config_path, 'r') as f:
        config = json.load(f)

    pprint(config)

    model, train_iterator, train_field = initialize(config)

    # val_iterator, f1 = create_iterator(f'_data/val_lines.txt', 1, config['device'], field=train_field)
    # test_iterator, f2 = create_iterator(f'_data/test_lines.txt', 1, config['device'], field=train_field)
    #
    # print(len(train_field.vocab), len(f1.vocab), len(f2.vocab))
    #
    # train(config, model, train_iterator, val_iterator)

    # nll, ppl, elbo, wpa = validate(config, model, test_iterator, phase='test')
    pprint(sample_sentences(config, model, train_field.vocab, max_len=20, n=10))
    pprint(reconstruct_sentence('I go to the cinema every Saturday', config, model, train_field.vocab))
