import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=RuntimeWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import argparse
import numpy as np
import scipy
from scipy.stats import norm
#our libs
import matplotlib.pyplot as plt
import seaborn as sns
from utils import *
from dataloader import *
from models import *
from gen_data import *
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DeepAR(nn.Module):
    def __init__(self, params):
        '''
        We define a recurrent network that predicts the future values of a time-dependent variable based on
        past inputs and covariates.
        '''
        super(DeepAR, self).__init__()
        self.params = args
        #self.embedding = nn.Embedding(args.num_class, 128)

        self.lstm = nn.LSTM(input_size=args.dims,#1+params.cov_dim+128,
                            hidden_size=128,
                            num_layers=2,
                            bias=True,
                            batch_first=False)
        '''self.lstm = nn.LSTM(input_size=1 + params.cov_dim,
                            hidden_size=params.lstm_hidden_dim,
                            num_layers=params.lstm_layers,
                            bias=True,
                            batch_first=False,
                            dropout=params.lstm_dropout)'''
        # initialize LSTM forget gate bias to be 1 as recommanded by http://proceedings.mlr.press/v37/jozefowicz15.pdf
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

        self.relu = nn.ReLU()
        self.distribution_mu = nn.Linear(128 * 2, args.dims)#1)
        self.distribution_presigma = nn.Linear(128 * 2, args.dims)#1)
        self.distribution_sigma = nn.Softplus()

    def forward(self, x, hidden=None, cell=None):
        '''
        Predict mu and sigma of the distribution for z_t.
        Args:
            x: ([1, batch_size, 1+cov_dim]): z_{t-1} + x_t, note that z_0 = 0
            idx ([1, batch_size]): one integer denoting the time series id
            hidden ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM h from time step t-1
            cell ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM c from time step t-1
        Returns:
            mu ([batch_size]): estimated mean of z_t
            sigma ([batch_size]): estimated standard deviation of z_t
            hidden ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM h from time step t
            cell ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM c from time step t
        '''
        #onehot_embed = self.embedding(idx) #TODO: is it possible to do this only once per window instead of per step?
        lstm_input = x.permute(1, 0, 2) #torch.cat((x, onehot_embed), dim=2)
        if hidden == None and cell == None:
            output, (hidden, cell) = self.lstm(lstm_input)
        else:
            output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        # use h from all three layers to calculate mu and sigma
        hidden_permute = hidden.permute(1, 2, 0).contiguous().view(hidden.shape[1], -1)
        pre_sigma = self.distribution_presigma(hidden_permute)
        mu = self.distribution_mu(hidden_permute)
        sigma = self.distribution_sigma(pre_sigma)  # softplus to make sure standard deviation is positive
        return torch.squeeze(mu), torch.squeeze(sigma), hidden, cell

    def init_hidden(self, input_size):
        return torch.zeros(2, input_size, 128, device=device)

    def init_cell(self, input_size):
        return torch.zeros(2, input_size, 128, device=device)

    def test(self, x, hidden=None, cell=None, sampling=False):
        #x = x.permute(1, 0, 2)
        print(x.shape)
        batch_size = x.shape[0]
        sample_times = 5
        if sampling:
            samples = torch.zeros(sample_times, batch_size, 1, args.dims,#self.params.predict_steps,
                                       device=device)
            for j in range(sample_times):#self.params.sample_times):
                decoder_hidden = hidden
                decoder_cell = cell
                for t in range(1):#self.params.predict_steps):
                    mu_de, sigma_de, decoder_hidden, decoder_cell = self(x)#, #[self.params.predict_start + t].unsqueeze(0),
                                                                         #decoder_hidden, decoder_cell)
                    gaussian = torch.distributions.normal.Normal(mu_de, sigma_de)
                    print(mu_de.shape, sigma_de.shape)
                    pred = gaussian.sample()  # not scaled
                    samples[j, :, t] = pred #pred * v_batch[:, 0] + v_batch[:, 1]
                    #if t < (self.params.predict_steps - 1):
                    #    x[self.params.predict_start + t + 1, :, 0] = pred

            sample_mu = torch.median(samples, dim=0)[0]
            sample_sigma = samples.std(dim=0)
            print("hello")
            return sample_mu.squeeze(1), sample_sigma.squeeze(1)#samples, sample_mu, sample_sigma

        else:
            decoder_hidden = hidden
            decoder_cell = cell
            sample_mu = torch.zeros(batch_size, self.params.predict_steps, device=self.params.device)
            sample_sigma = torch.zeros(batch_size, self.params.predict_steps, device=self.params.device)
            for t in range(self.params.predict_steps):
                mu_de, sigma_de, decoder_hidden, decoder_cell = self(x[self.params.predict_start + t].unsqueeze(0),
                                                                     decoder_hidden, decoder_cell)
                sample_mu[:, t] = mu_de * v_batch[:, 0] + v_batch[:, 1]
                sample_sigma[:, t] = sigma_de * v_batch[:, 0]
                if t < (self.params.predict_steps - 1):
                    x[self.params.predict_start + t + 1, :, 0] = mu_de
            return sample_mu, sample_sigma


def loss_fn(mu: Variable, sigma: Variable, labels: Variable):
    '''
    Compute using gaussian the log-likehood which needs to be maximized. Ignore time steps where labels are missing.
    Args:
        mu: (Variable) dimension [batch_size] - estimated mean at time step t
        sigma: (Variable) dimension [batch_size] - estimated standard deviation at time step t
        labels: (Variable) dimension [batch_size] z_t
    Returns:
        loss: (Variable) average log-likelihood loss across the batch
    '''
    #zero_index = (labels != 0)
    #print(mu.shape, sigma.shape, labels.shape)
    distribution = torch.distributions.normal.Normal(mu, sigma)#[zero_index], sigma[zero_index])
    likelihood = distribution.log_prob(labels)#[zero_index])
    return -torch.mean(likelihood)

def plot2d(Y, name):
    Y = Y.detach().cpu().numpy()
    sns.kdeplot(Y[:, 0], Y[:, 1], cmap='Blues', shade=True, thresh=0)
    plt.savefig("./" + name)
    plt.clf()

def histogram(Y, name):
    Y = Y.detach().cpu().numpy()
    plt.hist(Y, bins=25)
    plt.savefig("./" + name)
    plt.clf()

def plotaxis(Y, name):
    y1, y2 = Y[:,0], Y[:,1]
    histogram(y1, name=str(name)+'_x1.png')
    histogram(y2, name=str(name)+'_x2.png')

def gaussian_mixture(means, stds, p, args):
    assert np.sum(p) == 1
    k = len(p)
    ranges = [0.]
    for i in range(k):
        ranges.append(ranges[i] + p[i])
    mix = np.zeros((args.n, 1))
    idx = np.random.uniform(0, 1, size=(args.n, 1))
    for i in range(k):
        g = np.random.normal(loc=means[i], scale=stds[i], size=(args.n, 1))
        indices = np.logical_and(idx >= ranges[i], idx < ranges[i+1])
        mix[indices] = g[indices]
    return mix

def optimizer(net, args):
    assert args.optimizer.lower() in ["sgd", "adam", "radam"], "Invalid Optimizer"

    if args.optimizer.lower() == "sgd":
	       return optim.SGD(net.parameters(), lr=args.lr, momentum=args.beta1, nesterov=args.nesterov)
    elif args.optimizer.lower() == "adam":
	       return optim.Adam(net.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    elif args.optimizer.lower() == "radam":
           return radam.RAdam(net.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
 
def l1_quantile_loss(output, target, tau, reduce=True):
    u = target - output
    loss = (tau - (u.detach() <= 0).float()).mul_(u)
    return loss.mean() if reduce else loss

def huber_quantile_loss(output, target, tau, k=0.02, reduce=True):
    u = target - output
    loss = (tau - (u.detach() <= 0).float()).mul_(u.detach().abs().clamp(max=k).div_(k)).mul_(u)
    return loss.mean()

def test(net, args, name, loader):
    net.eval()
    X, Y = loader.dataset.getXY()
    U = torch.ones(X.shape[0], args.dims)*args.quantile
    U = U.cuda()
    X = X.cuda()
    Y_hat, sigma = net.test(X, sampling=True)
    print(Y_hat.shape)
    epsilon = torch.abs(Y_hat - Y)
    ql = l1_quantile_loss(Y_hat, Y, U)
    print('max : ' + str(epsilon.max().item()))
    print('mae : ' + str(epsilon.mean().item()))
    print('ql' + str(args.quantile*100) + ': ' + str(ql.item()))
    Y_hat, Y, U = Y_hat.detach().cpu().numpy(), Y.detach().cpu().numpy(), U.detach().cpu().numpy()
    sigma = sigma.detach().cpu().numpy()
    print("rmse: " + str(rmse(Y, Y_hat)))
    print("smape: " + str(smape(Y, Y_hat)))
    Y_hat = Y_hat + 1.28*sigma
    U = torch.ones(X.shape[0], args.dims)*0.9
    U = U.cuda()
    ql90 = l1_quantile_loss(torch.from_numpy(Y_hat).cuda(), torch.from_numpy(Y).cuda(), U.cuda())
    print("ql90: " + str(ql90.item()))

def validate(net, loader, args):
    net.eval()
    X, Y = loader.dataset.getXY()
    U = torch.ones(X.shape[0], args.dims)*args.quantile
    U = U.cuda()
    X = X.cuda()
    Y_hat = net(U, X)
    error = torch.abs(Y_hat - Y).mean()
    loss = huber_quantile_loss(Y_hat, Y, U)
    print("Val Loss : %.5f, Error : %.5f" % (loss.item(), error.item()))
    net.train()

def train(net, optimizer, loaders, args):
    train_loader, val_loader, test_loader = loaders
    for epoch in range(1, args.epoch+1):
        running_loss = 0.0
        for idx, (X, Y) in tqdm(enumerate(train_loader), total=len(train_loader)):
            U = torch.rand(size=(args.batch_size, args.dims))
            optimizer.zero_grad()
            U = U.cuda()
            hidden = net.init_hidden(args.batch_size)
            cell = net.init_cell(args.batch_size)
            mu, sigma, hidden, cell = net(X, hidden, cell)
            loss = loss_fn(mu, sigma, Y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print('%.5f' %
            (running_loss/args.iters))
        #validate(net, val_loader, args)
    test(net, args, name='imgs/trained.png', loader=test_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # optimization related arguments
    parser.add_argument('--batch_size', default=128, type=int,
                        help='input batch size')
    parser.add_argument('--epoch', default=10, type=int,
                        help='epochs to train for')
    parser.add_argument('--optimizer', default='adam', help='optimizer')
    parser.add_argument('--lr', default=0.005, type=float, help='LR')
    parser.add_argument('--beta1', default=0.9, type=float,
                        help='momentum for sgd, beta1 for adam')
    parser.add_argument('--beta2', default=0.999, type=float)
    parser.add_argument('--nesterov', default=False)
    parser.add_argument('--iters', default=1000, type=int)
    parser.add_argument('--mean', default=0, type=int)
    parser.add_argument('--std', default=1, type=int)
    parser.add_argument('--dims', default=2, type=int)
    parser.add_argument('--m', default=10, type=int)
    parser.add_argument('--n', default=2000, type=int)
    parser.add_argument('--quantile', default=0.5, type=float)
    parser.add_argument('--dataset', default='energy', type=str)
    args = parser.parse_args()

    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    if args.dataset == 'energy':
        args.dims = 28

    elif args.dataset == 'stock':
        args.dims = 6

    #trainloader, testloader = dataloader(args)
    net = DeepAR(args)
    ds = [TimeSeriesDataset(dataset=args.dataset, device=device, split=x) for x in ['train', 'val', 'test']]
    loaders = [data.DataLoader(d, batch_size=args.batch_size, shuffle=True, drop_last=True) for d in ds]
    optimizer = optimizer(net, args)
    net.to(device)
    train(net, optimizer, loaders, args)

    '''
    Y = torch.from_numpy(ds.y)
    plotaxis(Y, name='imgs/theor')
    plot2d(Y, name='imgs/theor.png')
    '''
    print("Training completed!")
