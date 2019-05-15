from project2.rnn.train import train


train_path = 'project_2_data/02-21.10way.clean'

if __name__ == '__main__':
    config = {
        'batch_size': 64,
        'bptt_len': 20,
        'learning_rate': 1e-3,
        'device': 'cpu',
    }
    corpus = train(config)

