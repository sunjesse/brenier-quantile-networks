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

class Synthetic(data.Dataset):
    def __init__(self, args,  n=1000):
        self.n = n
        np.random.seed(0)
        #self.y = np.random.normal(loc=0, scale=1, size=(self.n, 1))
        #self.y = np.random.multivariate_normal(mean=[2, 3], cov=np.array([[3,2],[2,5]]), size=(self.n))

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
        transform=transforms.Compose([transforms.ToTensor()])
        self.y_ = datasets.MNIST('../data', train=True, download=True,transform=transform)
        self.y = self.y_.data.flatten(-2, -1).float()/255.
        self.x = self.y_.targets

    def __len__(self):
        return len(self.y)#self.y

    def __getitem__(self, i):
        if torch.is_tensor(self.y):
            return self.y[i].float().to(device), self.x[i].to(device)
        y = torch.from_numpy(self.y[i]).float().to(device)
        x = torch.from_numpy(self.x[i]).to(device)
        return y, x

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
    U_ = unif(size=(100, 784))
    U = gauss.icdf(U_)
    a = torch.arange(0, 10, device=device)
    X = a*torch.ones((10, 10), device=device).long()
    X = X.permute(1, 0).flatten()
    print(X)
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
    #mnist
    print(Y_hat.shape)
    Y_hat = Y_hat.view(100, 28, 28).unsqueeze(1)
    utils.save_image(utils.make_grid(Y_hat, nrow=10),
        './mnist.png')
    return
    if args.dims == 1:
        histogram(Y_hat, name) # uncomment for 1d case
    else:
        plot2d(Y_hat, name='imgs/2d.png') # 2d contour plot
        plotaxis(Y_hat, name='imgs/train')


def train(net, optimizer, loader, args):
    k = args.k
    #eg = Rings()#EightGaussian()
    gauss = torch.distributions.normal.Normal(torch.tensor([0.]).cuda(), torch.tensor([1.]).cuda())
    for epoch in range(1, args.epoch+1):
        running_loss = 0.0
        for idx, (Y, label) in enumerate(loader):
            u = unif(size=(args.batch_size, 784))#args.dims))
            u = gauss.icdf(u)
            optimizer.zero_grad()
            alpha, beta = net(u)
            X = net.to_onehot(label)
            loss = dual(U=u, Y_hat=(alpha, beta), Y=Y, X=X, eps=args.eps)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        #if epoch % (args.epoch // 50) == 0:
        print('%.5f' %
            (running_loss/(idx+1)))

    test(net, args, name='imgs/trained.png', loader=loader)
    '''
    Y = eg.sample(5000).cuda()
    plotaxis(Y, name='imgs/theor')
    plot2d(Y, name='imgs/theor.png')
    '''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # optimization related arguments
    parser.add_argument('--batch_size', default=512, type=int,
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
    parser.add_argument('--dims', default=784, type=int)
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
    net = ConditionalConvexQuantile(xdim=10,
                                    args=args,
                                    a_hid=512,
                                    a_layers=3,
                                    b_hid=512,
                                    b_layers=1)

    net.apply(net.weights_init_uniform_rule)
    
    ds = Synthetic(args, n=args.n)
    loader = data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    optimizer = optimizer(net, args)
    net.cuda()
    train(net, optimizer, loader, args)
    #mnist
    #train(net, optimizer, loader, ds.y[:args.n].float().cuda(), args)

    if args.genTheor:
        Y = torch.from_numpy(ds.y)
        plotaxis(Y, name='imgs/theor')
        plot2d(Y, name='imgs/theor.png')

    print("Training completed!")
