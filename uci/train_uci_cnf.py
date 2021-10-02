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
from flows import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNF(nn.Module):
    def __init__(self, args, n_blocks, input_size, hidden_size, n_hidden, cond_label_size):
        super(CNF, self).__init__()
        self.flow = RealNVP(n_blocks=n_blocks,
                            input_size=input_size,
                            hidden_size=hidden_size,
                            n_hidden=n_hidden,
                            cond_label_size=cond_label_size)
        self.f = BiRNN(args.dims, hidden_size=args.dims*4, num_layers=2, xdim=cond_label_size)
    
    def forward(self, y, x):
        h = self.f(x)
        z, log_det = self.flow(y, h)
        return z, log_det

    def infer(self, x):
        h = self.f(x)
        z = self.flow.sample(cond=h)
        y, log_det = self.flow.inverse(z, h)
        return y, log_det

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
    Y_hat, _ = net.infer(X)
    print(Y_hat.shape)
    epsilon = torch.abs(Y_hat - Y)
    ql = l1_quantile_loss(Y_hat, Y, U)
    print('max : ' + str(epsilon.max().item()))
    print('mae : ' + str(epsilon.mean().item()))
    print('ql' + str(args.quantile*100) + ': ' + str(ql.item()))
    Y_hat, Y, U = Y_hat.detach().cpu().numpy(), Y.detach().cpu().numpy(), U.detach().cpu().numpy()
    print("rmse: " + str(rmse(Y, Y_hat)))
    print("smape: " + str(smape(Y, Y_hat)))
    #U = torch.ones(X.shape[0], args.dims)*0.9
    #U = U.cuda()
    #ql90 = l1_quantile_loss(torch.from_numpy(Y_hat).cuda(), torch.from_numpy(Y).cuda(), U.cuda())
    #print("ql90: " + str(ql90.item()))

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
    gaussian = torch.distributions.normal.Normal(loc=torch.zeros(args.dims).to(device), scale=torch.ones(args.dims).to(device))
    for epoch in range(1, args.epoch+1):
        running_loss = 0.0
        for idx, (X, Y) in tqdm(enumerate(train_loader), total=len(train_loader)):
            optimizer.zero_grad()
            z, log_det = net(Y, X)
            log_pz = gaussian.log_prob(z)#torch.sum(gaussian.log_prob(z), dim=1, keepdim=True)
            p_theta = log_pz + log_det
            loss = -torch.mean(p_theta)
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
    net = CNF(args=args, input_size=args.dims, n_blocks=3, n_hidden=2, hidden_size=128, cond_label_size=args.dims)
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
