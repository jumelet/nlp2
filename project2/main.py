import torch
from vae.train import initialize, train, test

if __name__ == '__main__':
    config = {
        'train_path': '/home/mariog/projects/nlp2/project2/data/02-21.10way.clean',
        'valid_path': '/home/mariog/projects/nlp2/project2/data/22.auto.clean',
        'test_path': '/home/mariog/projects/nlp2/project2/data/23.auto.clean',
        'batch_size': 64,
        'bptt_len': 40,
        # 'checkpoint': 'pickles/05-23_00:54/state_dict_e1.pt',
        'rnn_type': 'GRU',
        'device': 'cuda',
        'epochs': 20,
        'learning_rate': 1e-3,
        'hidden_dim': 191,
        'input_emb_dim': 353,
        'latent_dim': 13,
        'num_layers': 1,
        'bidir': True,
        'word_dropout_prob': 0.4,
        'importance_samples': 5,
        'seed': 0
    }

    model, vocab, train_iterator, valid_iterator, test_iterator = initialize(config)
    train(config, model, train_iterator, valid_iterator)
    test(config, model, test_iterator)
