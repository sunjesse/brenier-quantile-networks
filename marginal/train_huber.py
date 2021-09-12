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
import utils
from gsw import GSW
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from gen_data import *

class QNN(nn.Module):
    def __init__(self, args):
        super(QNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(args.dims, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, args.dims))
        	
    def forward(self, x, train=True):
        x = self.net(x)
        '''
        if train == False:
            with torch.no_grad():
                self.L_val.weight.copy_(self.L.weight)
                det = torch.det(self.L.weight)
                self.L_val.weight.div_(det**(1.0/args.dims))
        return x if train else self.L(x)
        ''' 
        return x #self.L(x)

    def get_zero_grad_hook(self, mask):
        def hook(grad):
            return grad * mask
        return hook


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
 
def gaussian_kernel(a, b):
    dim1_1, dim1_2 = a.shape[0], b.shape[0]
    depth = a.shape[1]
    a = a.view(dim1_1, 1, depth)
    b = b.view(1, dim1_2, depth)
    a_core = a.expand(dim1_1, dim1_2, depth)
    b_core = b.expand(dim1_1, dim1_2, depth)
    numerator = (a_core - b_core).pow(2).mean(2)/depth
    return torch.exp(-numerator)

def MMD(a, b):
    return gaussian_kernel(a, a).mean() + gaussian_kernel(b, b).mean() - 2*gaussian_kernel(a, b).mean()

def l1_quantile_loss(output, target, tau, reduce=True):
    u = target - output
    loss = (tau - (u.detach() <= 0).float()).mul_(u)
    return loss.mean() if reduce else loss

def huber_quantile_loss(output, target, tau, net, k=0.02, reduce=True):
    u = target - output
    loss = (tau - (u.detach() <= 0).float()).mul_(u.detach().abs().clamp(max=k).div_(k)).mul_(u)

    # covariance difference norm loss
    #cl =  torch.matmul(net.L.weight, net.L.weight.permute(1, 0)) - utils.cov(target.permute(-1,-2))
    #cl = torch.norm(cl, p=2)
    return loss.mean() #+ cl

def w_quantile_loss(output, target, tau, W, reduce=True):
    u = target - output
    loss = torch.matmul(u, W)*torch.matmul(tau - (u<0).float(), W.float())
    return loss.mean() if reduce else loss

def l2_loss(output, target, tau):
    # assume tau univariate
    u = target - output
    loss = torch.abs(u) + (2*tau - 1)*u
    loss = torch.norm(loss, p=2, dim=1)
    return loss.mean()

def test(net, args, name):
    net.eval()
    tsr = None
    with torch.no_grad():
        for i in range(100): # get args.batch_size x 100 samples
            #U = np.random.uniform(0, 1, size=(1, args.dims))
            #U = torch.from_numpy(U).float()
            U = torch.rand(size=(args.batch_size, args.dims))
            Y_hat = net(U, train=False)
            '''
        for idx, batch in enumerate(loader):
            Y, u = batch
            Y_hat = net(u)
            '''
            if tsr == None:
                tsr = Y_hat
            else:
                tsr = torch.cat([tsr, Y_hat], dim=0)
    if args.dims == 1:
        histogram(tsr, name) # uncomment for 1d case
    else:
        plot2d(tsr, name='imgs/2d.png') # 2d contour plot
        plotaxis(tsr, name='imgs/train')

def train(net, optimizer, loader, args):
    #loss_fn = GSW()
    k = args.k
    #W = utils.gen_random_projection(M=10, d=args.dims).permute(1, 0)
    #W = utils.linear(M=100, d=args.dims).permute(1, 0)
    for epoch in range(1, args.epoch+1):
        running_loss = 0.0
        for idx, batch in enumerate(loader):
            optimizer.zero_grad()
            loss = 0
            Y = batch
            #Y = torch.matmul(Y, W)
            for j in range(args.m):
                u = torch.rand(size=(args.batch_size, args.dims))
                Y_hat = net(u)
                '''
                if args.dims > 1:
                    Y_hat = torch.matmul(Y_hat, W)
                    #u = torch.matmul(u, W)
                '''
                #print(Y_hat.shape, Y.shape, u.shape)
                loss += l1_quantile_loss(Y_hat, Y, u) #MMD(Y_hat, Y)
            loss /= args.m
            loss.backward()
            optimizer.step()
            '''
            with torch.no_grad():
                net.L.weight.div_(torch.norm(net.L.weight, dim=1, keepdim=True))
            '''
            running_loss += loss.item()
        #print(net.L.weight)
        print('%.5f' %
        (running_loss/args.iters))
    test(net, args, name='imgs/trained.png')
    '''
    Y = np.random.multivariate_normal(mean=[0, 0], cov=np.array([[1,2],[2,1]]), size=(args.batch_size*100))
    Y = torch.from_numpy(Y)
    plotaxis(Y, name='imgs/theor')
    plot2d(Y, name='imgs/theor.png')
    '''

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
    parser.add_argument('--k', default=100, type=int)
    args = parser.parse_args()

    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    #trainloader, testloader = dataloader(args)
    net = QNN(args)
    ds = Synthetic(args, n=args.n)
    loader = data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    #net = torch.nn.RNN(input_size=1, hidden_size=1, num_layers=1, nonlinearity='relu')
    #criterion = nn.CrossEntropyLoss()
    optimizer = optimizer(net, args)
    train(net, optimizer, loader, args)
    #test(net, criterion, testloader, args)
    '''
    Y = torch.from_numpy(ds.y)
    plotaxis(Y, name='imgs/theor')
    plot2d(Y, name='imgs/theor.png')
    '''
    print("Training completed!")
