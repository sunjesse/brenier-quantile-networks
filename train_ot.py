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
from ot_modules.icnn import *
from gen_data import *
from torchvision import datasets, transforms, utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Synthetic(data.Dataset):
    def __init__(self, args,  n=1000):
        self.n = n
        np.random.seed(0)
        self.y = np.random.normal(loc=0, scale=1, size=(self.n, 1))
        #self.y = np.random.multivariate_normal(mean=[2, 3], cov=np.array([[3,2],[2,5]]), size=(self.n))

        #torch.manual_seed(0)
        #self.u = torch.rand(self.n, args.dims)
        #self.y = np.random.multivariate_normal(mean=[0, 0], cov=np.array([[1,0],[0,1]]), size=(self.n))

        #uniform
        #self.y1 = torch.rand(size=(self.n, 1))*5 + 1
        #self.y2 = torch.rand(size=(self.n, 1))*3 + 2
        #self.y = torch.cat([self.y1, self.y2], dim=1)

        #exp
        #self.y = np.random.exponential(scale=10, size=(self.n, 1))
        #self.y2 = np.random.exponential(scale=2, size=(self.n, 1))
        #self.y = np.concatenate([self.y1, self.y2], axis=1)
        
        #gaussian checkerboard
        #self.y = np.load('../data/synthetic/bary_ot_checkerboard_3.npy', allow_pickle=True).tolist()
        #self.y = self.y['Y']
       
        #spiral
        '''
        self.y, _ = make_spiral(n_samples_per_class=self.n, n_classes=1,
            n_rotations=2.5, gap_between_spiral=0.1, noise=0.2,
                gap_between_start_point=0.1, equal_interval=True)
        '''
        #self.y = np.random.standard_t(df=2, size=(args.n, 2))

        #mnist
        #transform=transforms.Compose([transforms.ToTensor()])
        #self.y_ = datasets.MNIST('../data', train=True, download=True,transform=transform)
        #self.y = self.y_.data.flatten(-2, -1).float()/255.

    def __len__(self):
        return 5000# len(self.y)#self.y

    def __getitem__(self, i):
        if torch.is_tensor(self.y):
            return self.y[i].cuda() / 255.
        return torch.from_numpy(self.y[i]).float().cuda()

class icq(nn.Module):
    def __init__(self, net, gs=True):
        super(icq, self).__init__()
        self.net = net
        self.gs = gs
        self.gauss = torch.distributions.normal.Normal(torch.tensor([0.]).cuda(), torch.tensor([1.]).cuda())

    def forward(self, u):
        if self.gs:
            U = self.gauss.icdf(u)
        else:
            U = u
            u = None
        return self.net(U)

    def invert(self, y):
        u = self.net.invert(y)
        if self.gs:
            u = self.gauss.cdf(u)
        return u
    
    def grad(self, u):
        if self.gs:
            U = self.gauss.icdf(u)
        else:
            U = u
        f = self.net(U).sum()
        Y_hat = torch.autograd.grad(f, U, create_graph=True)[0]
        return Y_hat
        
    def h(self, x):
        return torch.sqrt(1+3*x**2) + 2*x

    def ih(self, x):
        return -torch.sqrt(1+3*x**2) + 2*x

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

def test(net, args, name, loader, Y):
    net.eval()

    '''
    for p in list(net.parameters()):
        if hasattr(p, 'be_positive'):
            print(p)
    '''
    U = torch.rand(size=(args.n, args.dims), requires_grad=True).cuda()
    #U = torch.rand(size=(5000, 2), requires_grad=True).cuda()
    Y_hat = net.grad(U)
    print("max and min points generated: " + str(Y_hat.max()) + " " + str(Y_hat.min()))

    inverse = net.invert(Y_hat)
    m = (U - inverse).abs().max().item()
    print("max error of inversion: " + str(m))
    data = torch.sort(Y, dim=0)[0]
    z = net.invert(data)
    print("sampled points from target, sorted: " + str(data))
    print("corresponding quantiles: " + str(z))
    '''
    #mnist
    Y_hat = Y_hat.view(64, 28, 28).unsqueeze(1)
    utils.save_image(utils.make_grid(Y_hat),
        './mnist.png')
    return
    '''
    if args.dims == 1:
        histogram(Y_hat, name) # uncomment for 1d case
    else:
        plot2d(Y_hat, name='imgs/2d.png') # 2d contour plot
        plotaxis(Y_hat, name='imgs/train')

positive_params = []

def dual(U, Y_hat, Y, eps=0):
    loss = torch.mean(Y_hat)
    Y = Y.permute(1, 0)
    psi = torch.mm(U, Y) - Y_hat
    sup, _ = torch.max(psi, dim=0)
    loss += torch.mean(sup)

    if eps == 0:
        return loss

    l = torch.exp((psi-sup)/eps)
    loss += eps*torch.mean(l)
    return loss

def train(net, optimizer, loader, ds, args):
    k = args.k
    print(ds.shape)
    print(ds.min(), ds.max())
    #eg = Rings()#EightGaussian()
    for epoch in range(1, args.epoch+1):
        
        #for idx, Y in enumerate(loader):
        #u = torch.rand(size=(args.batch_size, args.dims)).cuda()
        #Y = eg.sample(5000).cuda()
        u = torch.rand(size=(args.n, args.dims)).cuda()
        running_loss = 0.0
        optimizer.zero_grad()
        Y_hat = net(u)
        loss = dual(U=u, Y_hat=Y_hat, Y=ds, eps=args.eps)
        loss.backward()
        optimizer.step()

        for p in positive_params:
            p.data.copy_(torch.relu(p.data))
        running_loss += loss.item()
        if epoch % (args.epoch // 50) == 0:
            print('%.5f' %
            (running_loss/args.iters))
    test(net, args, name='imgs/trained.png', loader=loader, Y=ds)
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
    parser.add_argument('--epoch', default=100, type=int,
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
    net_ = ICNN_LastInp_Quadratic(input_dim=args.dims,
                                 hidden_dim=512,#1024,#512
                                 activation='celu',
                                 num_layer=3)
    net = icq(net_, gs=args.gaussian_support)
    for p in list(net.parameters()):
        if hasattr(p, 'be_positive'):
            positive_params.append(p)
        p.data = torch.from_numpy(truncated_normal(p.shape, threshold=1./np.sqrt(p.shape[1] if len(p.shape)>1 else p.shape[0]))).float()

    ds = Synthetic(args, n=args.n)
    loader = data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    optimizer = optimizer(net, args)
    net.cuda()
    #train(net, optimizer, torch.from_numpy(ds.y).float().cuda(), args)
    train(net, optimizer, loader, torch.from_numpy(ds.y).float().cuda(), args)
    #mnist
    #train(net, optimizer, loader, ds.y[:args.n].float().cuda(), args)

    if args.genTheor:
        Y = torch.from_numpy(ds.y)
        plotaxis(Y, name='imgs/theor')
        plot2d(Y, name='imgs/theor.png')

    print("Training completed!")
