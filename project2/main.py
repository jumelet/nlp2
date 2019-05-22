from rnn.train import train


train_path = 'project_2_data/02-21.10way.clean'

if __name__ == '__main__':
    config = {
        'batch_size': 64,
        'device': 'cpu',
        'epochs': 10,
        'learning_rate': 1e-3,
        'hidden_size': 256,
        'input_emb_dim': 256,
        'num_layers': 1,
        'sample_every': 10,
        'sample_greedy': False,
        'temperature': 0.5,
    }
    corpus = train(config)

