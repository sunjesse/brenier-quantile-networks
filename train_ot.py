import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=RuntimeWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils import data
import argparse
import numpy as np
import scipy
from scipy.stats import norm
#our libs
from lib import radam
import matplotlib.pyplot as plt
import seaborn as sns
import utils
from utils import truncated_normal
from gsw import GSW
from ot_modules.icnn import *
from ot_modules.dual import dual

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Synthetic(data.Dataset):
    def __init__(self, args,  n=1000):
        self.n = n
        np.random.seed(0)
        self.y = np.random.normal(loc=0, scale=1, size=(self.n, 2))
        #self.y = np.random.multivariate_normal(mean=[2, 3], cov=np.array([[3,-2],[-2,5]]), size=(self.n))

        #torch.manual_seed(0)
        #self.u = torch.rand(self.n, args.dims)
        #self.y = np.random.multivariate_normal(mean=[0, 0], cov=np.array([[1,0],[0,1]]), size=(self.n))

        #uniform
        #self.y1 = torch.rand(size=(self.n, 1))*5 + 1
        #self.y2 = torch.rand(size=(self.n, 1))*3 + 2
        #self.y = torch.cat([self.y1, self.y2], dim=1)

        #exp
        #self.y1 = np.random.exponential(scale=10, size=(self.n, 1))
        #self.y2 = np.random.exponential(scale=2, size=(self.n, 1))
        #self.y = np.concatenate([self.y1, self.y2], axis=1)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        #x = torch.empty(args.dims)
        #x[self.u[i] < 0.5] = 0.5
        #x[self.u[i] >= 0.5] = 0.75
        #u = torch.cat([x, self.u[i]], dim=-1).float()
        return torch.from_numpy(self.y[i]).float()#, self.u[i].float()#, torch.from_numpy(self.x[i]).float()

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
        '''
        self.L = nn.Linear(args.dims, args.dims)
        with torch.no_grad():
            self.L.weight.copy_(torch.eye(args.dims))
        mask = torch.tril(torch.ones(args.dims, args.dims))
        self.L.weight.register_hook(self.get_zero_grad_hook(mask))
        self.L_val = nn.Linear(args.dims, args.dims)
        '''
       # with torch.no_grad():
       #     self.L.weight.div_(torch.norm(self.L.weight, dim=1, keepdim=True))
        	
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

def test(net, args, name, loader):
    net.eval()

    '''
    for p in list(net.parameters()):
        if hasattr(p, 'be_positive'):
            print(p)
    '''
    U = torch.rand(size=(2000, args.dims), requires_grad=True)
    f = net(U).sum()
    Y_hat = torch.autograd.grad(f, U, create_graph=True)[0]

    if args.dims == 1:
        histogram(Y_hat, name) # uncomment for 1d case
    else:
        plot2d(Y_hat, name='imgs/2d.png') # 2d contour plot
        plotaxis(Y_hat, name='imgs/train')

positive_params = []

def train(net, optimizer, loader, args):
    #loss_fn = GSW()
    k = args.k
    #W = utils.gen_random_projection(M=10, d=args.dims).permute(1, 0)
    #W = utils.linear(M=100, d=args.dims).permute(1, 0)
    for epoch in range(1, args.epoch+1):
        running_loss = 0.0
        for idx, batch in enumerate(loader):
            if idx < len(loader)-1:
                optimizer.zero_grad()
            loss = 0
            Y = batch
            #Y = torch.matmul(Y, W)
            U = torch.empty(size=(args.batch_size, args.dims, args.m))
            Y_hats = torch.empty(size=(args.batch_size, 1, args.m))
            for j in range(args.m):
                u = torch.rand(size=(args.batch_size, args.dims))
                U[:, :, j] = u
                Y_hat = net(u)
                Y_hats[:, :, j] = Y_hat
            loss += dual(U=U, Y_hat=Y_hats, Y=Y)
            #loss /= args.m
            loss.backward()
            '''
            for p in list(net.parameters()):
                p.grad.copy_(-p.grad)
            '''
            optimizer.step()

            for p in positive_params:
                p.data.copy_(torch.relu(p.data))
            running_loss += loss.item()
        #print(net.L.weight)
        print('%.5f' %
        (running_loss/args.iters))
    test(net, args, name='imgs/trained.png', loader=loader)
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
    parser.add_argument('--n', default=5000, type=int)
    parser.add_argument('--k', default=100, type=int)
    args = parser.parse_args()

    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    #trainloader, testloader = dataloader(args)
    net = ICNN_LastInp_Quadratic(input_dim=args.dims,
                                 hidden_dim=128,
                                 activation='celu',
                                 num_layer=2)

    for p in list(net.parameters()):
        if hasattr(p, 'be_positive'):
            positive_params.append(p)
        p.data = torch.from_numpy(truncated_normal(p.shape, threshold=1./np.sqrt(p.shape[1] if len(p.shape)>1 else p.shape[0]))).float()

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
