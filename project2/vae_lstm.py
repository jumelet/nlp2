from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from preprocess_data import Data_reader

parser = argparse.ArgumentParser(description='VAE lm')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

# load training and validation data
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


data_loader_train = Data_reader('project_2_data/23.auto.clean')
data_loader_valid = Data_reader('project_2_data/22.auto.clean')


class VAE_LM(nn.Module):
    def __init__(self, hidden_dim, embedding_dim, vocab_size, z_dim):
        super(VAE_LM, self).__init__()

        self.embeds = nn.Embedding(vocab_size, embedding_dim)

        self.encoder = nn.LSTM(embedding_dim, hidden_dim)
        self.dense_mu = nn.Linear(hidden_dim, z_dim)
        self.dense_sigma = nn.Linear(hidden_dim, z_dim)

        self.affine_sample = nn.Linear(z_dim, hidden_dim)
        self.decoder = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden_to_vocab_linear = nn.Linear(hidden_dim, vocab_size)

        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def reparameterize(self, mu, sigma):
        eps = torch.randn_like(sigma)
        return mu + eps * sigma

    def forward(self, sentences_batch):
        log_softmax = nn.LogSoftmax(dim=1)
        rnn_input = nn.utils.rnn.pack_sequence([s[:-1] for s in sentences_batch])
        rnn_target = nn.utils.rnn.pack_sequence([s[1:] for s in sentences_batch])

        embeddings_batch = nn.utils.rnn.PackedSequence(self.embeds(rnn_input.data), rnn_input.batch_sizes)

        _, hidden_encoded = self.encoder(embeddings_batch)
        hidden_encoded = hidden_encoded[0].squeeze(0)

        mu = self.dense_mu(hidden_encoded)
        sigma = self.softplus(self.dense_sigma(hidden_encoded))

        z = self.reparameterize(mu, sigma)
        z = self.tanh(self.affine_sample(z))

        # hidden state has to be provided together will cell state as a tuple (h,c)
        outputs, hidden_decoded = self.decoder(embeddings_batch, (z.unsqueeze(0), torch.zeros(z.size()).unsqueeze(0)))

        hidden_decoded = hidden_decoded[0].squeeze(0)
        # process outputs
        out = self.hidden_to_vocab_linear(outputs.data)

        out = log_softmax(out)
        return out, mu, sigma, rnn_target


model = VAE_LM(191, 353, len(data_loader_train.i2w), 13).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def kl(mu, sigma):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # KL divergence
    z_var = sigma ** 2

    kl_z = -0.5 * torch.sum(1 + torch.log(z_var) - mu ** 2 - z_var, 1)

    return kl_z


def train(epoch):

    nll_loss = nn.NLLLoss()

    model.train()
    train_loss = 0

    for batch_idx, batch in enumerate(data_loader_train.create_batches(args.batch_size)):

        optimizer.zero_grad()
        softmax_out, mu, sigma, rnn_target = model(batch)
        kl_z = kl(mu, sigma)
        nll_p_x_z = nll_loss(softmax_out, rnn_target.data)

        loss = nll_p_x_z + torch.mean(kl_z)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {}, Loss: {}'.format(
                epoch, train_loss))

    # print('====> Epoch: {} Average loss: {:.4f}'.format(
     #     epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                           'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        # test(epoch)
        # with torch.no_grad():
        #    sample = torch.randn(64, 20).to(device)
        #    sample = model.decode(sample).cpu()
        #    save_image(sample.view(64, 1, 28, 28),
        #              'results/sample_' + str(epoch) + '.png')
