import json
from vae.train import initialize, train, test
from vae.analysis import sample_sentences, reconstruct_sentence

if __name__ == '__main__':

    # with open('vae/configs/lstm.config', 'w') as f:
    #     json.dump(config, f, indent=4)

    with open('vae/configs/lstm.config', 'r') as f:
        config = json.load(f)

    model, vocab, train_iterator, valid_iterator, test_iterator = initialize(config)
    train(config, model, train_iterator, valid_iterator, vocab)
    test(config, model, test_iterator, vocab)
    # print(sample_sentences(config, model, vocab, max_len=20, n=10))
    # print(reconstruct_sentence('I go to the cinema every Saturday', config, model, vocab))