import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=RuntimeWarning)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import argparse
import numpy as np
import scipy
import scipy.stats as ss
from scipy.stats import norm
#our libs
import matplotlib.pyplot as plt
import seaborn as sns
from utils import *
from gen_data import *
from torchvision import datasets, transforms, utils
from models import *
from supp.distribution_output import *
from supp.piecewise_linear import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)

def gen_gaussian_mixture(means, stds, p, n):
    np.random.seed(0)
    assert np.sum(p) == 1
    k = len(p)
    ranges = [0.]
    for i in range(k):
        ranges.append(ranges[i] + p[i])
    mix = np.zeros((n, 1))
    idx = np.random.uniform(0, 1, size=(n, 1))
    for i in range(k):
        g = np.random.normal(loc=means[i], scale=stds[i], size=(n, 1))
        indices = np.logical_and(idx >= ranges[i], idx < ranges[i+1])
        mix[indices] = g[indices]
    return mix

class Synthetic(data.Dataset):
    def __init__(self, args):
        self.n = args.n 
        np.random.seed(0)
        self.y = gen_gaussian_mixture(means=[-3., 0., 3.], stds=[.4, .4, .4], p=[.3, .4, .3], n=self.n)
        self.x = gen_gaussian_mixture(means=[-3., 0., 3.], stds=[.4, .4, .4], p=[.3, .4, .3], n=24*self.n)
        self.x = np.reshape(self.x, (self.n, 24))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if torch.is_tensor(self.y):
            return self.y[i].float().to(device)
        y = torch.from_numpy(self.y[i]).float().to(device)
        x = torch.from_numpy(self.x[i]).float().to(device)
        return x.unsqueeze(-1), y

def mix_norm_cdf(weights, means, covars, n=10000, plot=False):
    np.random.seed(0)
    means = np.expand_dims(np.array(means), 1)
    covars = np.expand_dims(np.array(covars), 1)
    params = np.concatenate([means, covars], 1)
    mixture_idx = np.random.choice(len(weights), size=n, replace=True, p=weights)
    y = np.fromiter((ss.norm.rvs(*(params[i])) for i in mixture_idx), dtype=np.float64)
	
    xs = np.linspace(y.min(), y.max(), n)
    ys = np.zeros_like(xs)
    for (l, s), w in zip(params, weights):
        ys += ss.norm.cdf(xs, loc=l, scale=s) * w
    if plot:
        plt.plot(ys, xs, label='Theoretical')
    else:
        return ys, xs

def plotaxis(Y1, Y2, Y3):
    f, axs = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
    axs[2].plot(np.linspace(0, 1, len(Y1), endpoint=False), Y1, label='OT (Ours)')
    axs[1].plot(np.linspace(0, 1, len(Y2), endpoint=False), Y2, label='IQN')
    axs[0].plot(np.linspace(0, 1, len(Y3), endpoint=False), Y3, label='Spline RNN')
    #plt.plot(X, Y1, label='OT (Ours)')
    #plt.plot(X, Y2, label='Huber')
    ys, xs = mix_norm_cdf(weights=[.3, .4, .3], means=[-3., 0., 3.], covars=[.4, .4, .4])
    axs[0].plot(ys, xs, '--', label='Theoretical')
    axs[1].plot(ys, xs, '--', label='Theoretical')
    axs[2].plot(ys, xs, '--', label='Theoretical')
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    plt.savefig('./quantiles.png')	

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

def huber_quantile_loss(output, target, tau, k=0.02, reduce=True):
    u = target - output
    loss = (tau - (u.detach() <= 0).float()).mul_(u.detach().abs().clamp(max=k).div_(k)).mul_(u)
    return loss.mean()

def optimizer(net, args):
    assert args.optimizer.lower() in ["sgd", "adam", "radam"], "Invalid Optimizer"

    if args.optimizer.lower() == "sgd":
	       return optim.SGD(net.parameters(), lr=args.lr, momentum=args.beta1, nesterov=args.nesterov)
    elif args.optimizer.lower() == "adam":
	       return optim.Adam(net.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

def unif(size, eps=1E-7):
    return torch.clamp(torch.rand(size).cuda(), min=eps, max=1-eps)

def test(net1, net2, net3, args):
    net1.eval() # OT
    net2.eval() # Huber QL
    net3.eval()
    gauss = torch.distributions.normal.Normal(torch.tensor([0.]).cuda(), torch.tensor([1.]).cuda())
    U_s, Y = mix_norm_cdf(weights=[.3, .4, .3], means=[-3., 0., 3.], covars=[.4, .4, .4], n=1000, plot=False)
    U_ = unif(size=(1000, 1))
    #U_ = torch.from_numpy(U_).float().cuda().unsqueeze(-1)
   # U_ = torch.clamp(U_, min=1E-7, max=1-(1E-7))
    print(U_.min(), U_.max())
    U = gauss.icdf(U_)

    X = gen_gaussian_mixture(means=[-3., 0., 3.], stds=[.4, .4, .4], p=[.3, .4, .3], n=24*1000)
    X = torch.tensor(np.reshape(X, (1000, 24)), dtype=torch.float32).cuda().unsqueeze(-1)
    Y_hat1 = net1.grad(U, X, onehot=False)#= net.forward(U, grad=True).sum()
    Y_hat2 = net2(U_, X)
    Y_hat3 = net3(x=X, u=U_).unsqueeze(-1)
    print("[net 1] max and min points generated: " + str(Y_hat1.max()) + " " + str(Y_hat1.min()))
    print("[net 2] max and min points generated: " + str(Y_hat2.max()) + " " + str(Y_hat2.min()))
    print("[net 3] max and min points generated: " + str(Y_hat3.max()) + " " + str(Y_hat3.min()))
    Y1 = np.sort(Y_hat1.squeeze(1).detach().cpu().numpy())
    Y2 = np.sort(Y_hat2.squeeze(1).detach().cpu().numpy())
    Y3 = np.sort(Y_hat3.squeeze(1).detach().cpu().numpy())
    plotaxis(Y1, Y2, Y3)
    eps1 = np.abs(Y - Y1)
    eps2 = np.abs(Y - Y2)
    eps3 = np.abs(Y - Y3)
    print("OT:")
    print("MeanAE: " + str(eps1.mean()))
    print("MaxAE: " + str(eps1.max()))
    print("sMAPE: " + str(smape(Y, Y1)))
    print("RMSE: " + str(rmse(Y, Y1)))
    print("QL: " + str(quantile_loss(Y, Y1, U_s)))
    
    print("IQN:")
    print("MeanAE: " + str(eps2.mean()))
    print("MaxAE: " + str(eps2.max()))
    print("sMAPE: " + str(smape(Y, Y2)))
    print("RMSE: " + str(rmse(Y, Y2)))
    print("QL: " + str(quantile_loss(Y, Y2, U_s)))

    print("Spline:")
    print("MeanAE: " + str(eps3.mean()))
    print("MaxAE: " + str(eps3.max()))
    print("sMAPE: " + str(smape(Y, Y3)))
    print("RMSE: " + str(rmse(Y, Y3)))
    print("QL: " + str(quantile_loss(Y, Y3, U_s)))

def train(net, optimizer, loader, args, marginal=False):
    k = args.k
    gauss = torch.distributions.normal.Normal(torch.tensor([0.]).cuda(), torch.tensor([1.]).cuda())
    for epoch in range(1, args.epoch+1):
        running_loss = 0.0
        for idx, (X, Y) in enumerate(loader):
            u = unif(size=(args.batch_size, 1))#args.dims))
            optimizer.zero_grad()
            if marginal == True:
                Y_hat = net(u, X)
                loss = huber_quantile_loss(Y_hat, Y, u)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                continue
            u = gauss.icdf(u)
            X = net.f(X)
            alpha, beta = net(u)
            loss = dual(U=u, Y_hat=(alpha, beta), Y=Y, X=X, eps=args.eps)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if epoch % 10 == 0:
            print('%.5f' %
                (running_loss/(idx+1)))

def train_spline(net, optimizer, loader, args):
    k = args.k
    gauss = torch.distributions.normal.Normal(torch.tensor([0.]).cuda(), torch.tensor([1.]).cuda())
    for epoch in range(1, args.epoch+1):
        running_loss = 0.0
        for idx, (X, Y) in enumerate(loader):
            optimizer.zero_grad()
            loss = net(X, Y).mean()
            #print(loss.shape, X.shape, Y.shape)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if epoch % 10 == 0:
            print('%.5f' %
                (running_loss/(idx+1)))

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
    parser.add_argument('--dims', default=1, type=int)
    parser.add_argument('--m', default=10, type=int)
    parser.add_argument('--n', default=10000, type=int)
    parser.add_argument('--k', default=100, type=int)
    parser.add_argument('--genTheor', action='store_true')
    parser.add_argument('--gaussian_support', action='store_true')
    parser.add_argument('--eps', default=0, type=float)
    args = parser.parse_args()

    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    torch.cuda.set_device('cuda:0')

    net = ConditionalConvexQuantile(xdim=1, #OT
                                    a_hid=50,
                                    a_layers=3,
                                    b_hid=50,
                                    b_layers=1,
                                    args=args)
    
    '''
    net2 = nn.Sequential(nn.Linear(1, 50), #Huber QL
                         nn.BatchNorm1d(50),
                         nn.CELU(inplace=True),
                         nn.Linear(50, 50),
                         nn.BatchNorm1d(50),
                         nn.CELU(inplace=True),
                         nn.Linear(50, 1))
    '''
    net2 = IQN(args)
    net3 = Spline(args)

    ds = Synthetic(args)
    loader = data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    optimizer1 = optimizer(net, args)
    net.cuda()
    train(net, optimizer1, loader, args)

    optimizer2 = optimizer(net2, args)
    net2.cuda()
    train(net2, optimizer2, loader, args, marginal=True)

    optimizer3 = optimizer(net3, args)
    net3.cuda()
    train_spline(net3, optimizer3, loader, args)
    test(net, net2, net3, args)

    print("Training completed!")
