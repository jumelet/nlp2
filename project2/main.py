from rnn.train import train


if __name__ == '__main__':
    config = {
        'batch_size': 64,
        'bptt_len': 30,
        # 'checkpoint': 'pickles/05-23_21:12/state_dict_e10.pt',
        'device': 'cpu',
        'dropout': 0.5,
        'epochs': 20,
        'learning_rate': 1e-3,
        'hidden_size': 256,
        'input_emb_dim': 256,
        'num_layers': 1,
        'sample_every': 5,
        'sample_greedy': False,
        'save_every': -1,
        'temperature': 1.,
        'tie_weights': True,
    }
    corpus = train(config)

