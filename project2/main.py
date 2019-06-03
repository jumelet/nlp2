import argparse
import json
from vae.train import initialize, train, test
from vae.analysis import sample_sentences, reconstruct_sentence


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Sentence VAE')
    parser.add_argument('-c', '--config', help='Config dictionary in a txt file', required=True)
    config_path = parser.parse_args().config

    with open(config_path, 'r') as f:
        config = json.load(f)

    model, vocab, train_iterator, valid_iterator, test_iterator = initialize(config)
    train(config, model, train_iterator, valid_iterator, vocab)
    test(config, model, test_iterator, vocab)
    # print(sample_sentences(config, model, vocab, max_len=20, n=10))
    # print(reconstruct_sentence('I go to the cinema every Saturday', config, model, vocab))
