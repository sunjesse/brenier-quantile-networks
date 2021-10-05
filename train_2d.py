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
from utils import truncated_normal
from gen_data import *
from torchvision import datasets, transforms, utils
from models import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)

class Synthetic(data.Dataset):
    def __init__(self, args,  n=1000):
        self.n = n
        np.random.seed(0)
        #self.y = np.random.normal(loc=0, scale=1, size=(self.n, 1))
        #self.y = np.random.multivariate_normal(mean=[2, 3], cov=np.array([[3,2],[2,5]]), size=(self.n))

        #torch.manual_seed(0)
        self.y, self.x = make_spiral(n_samples_per_class=self.n, n_classes=3,
            n_rotations=1.5, gap_between_spiral=1.0, noise=0.2,
                gap_between_start_point=0.1, equal_interval=True)
        '''

        self.y, self.x = make_moons(n_samples=args.n, xy_ratio=2.0, x_gap=-0.2, y_gap=0.2, noise=0.1)
        '''

    def __len__(self):
        return len(self.y)#self.y

    def __getitem__(self, i):
        if torch.is_tensor(self.y):
            return self.y[i].float().to(device), self.x[i].to(device)
        y = torch.from_numpy(self.y[i]).float().to(device)
        x = torch.from_numpy(np.array(self.x[i])).to(device)
        return y, x

def plot2d(Y, name, labels=None):
    Y = Y.detach().cpu().numpy()
    #labels = labels.detach().cpu().numpy().flatten()
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    #sns.kdeplot(Y[:, 0], Y[:, 1], cmap='Blues', shade=True, thresh=0)
    sns.scatterplot(x=Y[:,0], y=Y[:,1], hue=labels, s=5)
    '''
    H, _, _ = np.histogram2d(Y[:, 0], Y[:, 1], 200, range=[[-4, 4], [-4, 4]])
    plt.imshow(H.T, cmap='BuPu')
    '''
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    plt.axis('off')
    plt.tight_layout()
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

def unif(size, eps=1E-7):
    return torch.clamp(torch.rand(size).cuda(), min=eps, max=1-eps)

def test(net, args, name, loader):
    net.eval()

    '''
    for p in list(net.parameters()):
        if hasattr(p, 'be_positive'):
            print(p)
    '''
    gauss = torch.distributions.normal.Normal(torch.tensor([0.]).cuda(), torch.tensor([1.]).cuda())
    U = unif(size=(5000, 2))
    U = gauss.icdf(U)
    X = torch.zeros(5000, device=device).long()
    X[:5000//3] = 1
    X[5000//3: 10000//3] = 2
    #print(X)
    Y_hat = net.grad(U, X)#= net.forward(U, grad=True).sum()
    #Y_hat = net.grad(U)
    print("max and min points generated: " + str(Y_hat.max()) + " " + str(Y_hat.min()))
    '''

    inverse = net.invert(Y_hat)
    m = (U - inverse).abs().max().item()
    print("max error of inversion: " + str(m))
    data = torch.sort(Y, dim=0)[0]
    z = net.invert(data)
    z = gauss.cdf(z)
    print("sampled points from target, sorted: " + str(data))
    print("corresponding quantiles: " + str(z))
    '''
    print(Y_hat.shape)
    plot2d(Y_hat, name='imgs/2d.png', labels=X.cpu().numpy()) # 2d contour plot
    #plotaxis(Y_hat, name='imgs/train')

positive_params = []

def train(net, optimizer, loader, args):
    k = args.k
    #eg = Rings() # EightGaussian()
    gauss = torch.distributions.normal.Normal(torch.tensor([0.]).cuda(), torch.tensor([1.]).cuda())
    for epoch in range(1, args.epoch+1):
        running_loss = 0.0
        for idx, (Y, label) in enumerate(loader):
            #Y = eg.sample(args.batch_size).cuda()
            u = unif(size=(args.batch_size, 2))#args.dims))
            u = gauss.icdf(u)
            optimizer.zero_grad()
            #label[:] = 0
            #label[:args.batch_size//2] = 1
            X = net.to_onehot(label)
            alpha, beta= net(u)
            loss = dual(U=u, Y_hat=(alpha, beta), Y=Y, X=X, eps=args.eps)
            loss.backward()
            optimizer.step()
            #for p in positive_params:
            #    p.data.copy_(torch.relu(p.data))
            running_loss += loss.item()
        if epoch % (args.epoch//20) == 0:
            print('%.5f' %
                (running_loss))#/(idx+1)))

    test(net, args, name='imgs/trained.png', loader=loader)
    '''
    Y = torch.tensor(loader.dataset.y)#eg.sample(1000).cuda()
    X = loader.dataset.x
    plot2d(Y, labels=X, name='imgs/theor.png')
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
    parser.add_argument('--genTheor', action='store_true')
    parser.add_argument('--gaussian_support', action='store_true')
    parser.add_argument('--eps', default=0, type=float)
    args = parser.parse_args()

    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    torch.cuda.set_device('cuda:0')
    net = ConditionalConvexQuantile(xdim=3, 
                                    a_hid=512,
                                    a_layers=3,
                                    b_hid=512,
                                    b_layers=3,
                                    args=args)

    #for p in list(net.parameters()):
    #    if hasattr(p, 'be_positive'):
    #        positive_params.append(p)
    #    p.data = torch.from_numpy(truncated_normal(p.shape, threshold=1./np.sqrt(p.shape[1] if len(p.shape)>1 else p.shape[0]))).float()

    ds = Synthetic(args, n=args.n)
    loader = data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    optimizer = optimizer(net, args)
    net.cuda()
    train(net, optimizer, loader, args)
    #mnist
    #train(net, optimizer, loader, ds.y[:args.n].float().cuda(), args)

    if args.genTheor:
        Y = torch.from_numpy(ds.y)
        #plotaxis(Y, name='imgs/theor')
        plot2d(Y, labels=ds.x, name='imgs/theor.png')

    print("Training completed!")
