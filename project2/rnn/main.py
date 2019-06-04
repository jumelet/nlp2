from rnn.train import train


if __name__ == '__main__':
    config = {
        'batch_size': 64,
        # 'bptt_len': 30,
        'checkpoint': 'state_dict_e3_i550_p151.pt',
        'device': 'cpu',
        'dropout': 0.5,
        'epochs': 20,
        'learning_rate': 1e-3,
        'hidden_size': 337,
        'input_emb_dim': 464,
        'num_layers': 1,
        'sample_every': 1,
        'sample_greedy': False,
        'save_every': -1,
        'temperature': 1.,
        'tie_weights': False,
    }
    corpus = train(config)

