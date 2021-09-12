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
from utils import load, save, truncated_normal
from gen_data import *
from torchvision import datasets, transforms, utils
from models import *
from dataloader import *
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)

def plot2d(Y, name, labels=None):
    Y = Y.detach().cpu().numpy()
    #labels = labels.detach().cpu().numpy().flatten()
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    #sns.kdeplot(Y[:, 0], Y[:, 1], cmap='Blues', shade=True, thresh=0)
    sns.scatterplot(x=Y[:,0], y=Y[:,1], hue=labels)
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

def test(net, args, name, loader, label=None):
    net.eval()

    X, Y = loader.dataset.getXY()
    #X = label.expand(1, label.shape[-1])
    #X = X.repeat(9, 1)
    gauss = torch.distributions.normal.Normal(torch.tensor([0.]).cuda(), torch.tensor([1.]).cuda())
    #denom = torch.arange(1, 10)
    #U = (torch.ones(9, 2)/10.)*denom.unsqueeze(-1)#*0.5
    U = torch.ones(X.shape[0], args.dims)*args.quantile
    U = U.cuda()
    U = gauss.icdf(U)
    X = X.cuda()
    #X = torch.zeros(1000, device=device).long()
    #print(X)
    Y_hat = net.grad(U, X, onehot=False)#= net.forward(U, grad=True).sum()
    epsilon = torch.abs(Y_hat - Y)
    print(epsilon)
    print('max : ' + str(epsilon.max()))
    #Y_hat = net.grad(U)
    print('mae : ' + str(epsilon.mean()))
    print("max and min points generated: " + str(Y_hat.max()) + " " + str(Y_hat.min()))
    #plot2d(Y_hat, name='imgs/2d.png', labels=X.cpu().numpy()) # 2d contour plot
    #plotaxis(Y_hat, name='imgs/train')

def validate(net, loader, args):
    net.eval()
    gauss = torch.distributions.normal.Normal(torch.tensor([0.]).cuda(), torch.tensor([1.]).cuda())
    X, Y = loader.dataset.getXY()
    #u = unif(size=(X.shape[0], args.dims))
    u = torch.ones(size=(X.shape[0], args.dims))*args.quantile
    u = u.cuda()
    u = gauss.icdf(u)
    X_ = net.f(X)#bn1(X)
    alpha, beta = net(u)
    loss = dual(U=u, Y_hat=(alpha, beta), Y=Y, X=X_, eps=args.eps)
    Y_hat = net.grad(u, X, onehot=False)
    #error = F.mse_loss(Y_hat, Y, reduction='mean')
    error = torch.abs(Y_hat - Y).mean()#torch.sum((Y_hat - Y)**2/(Y_hat.shape[0]*Y_hat.shape[1]))
    print("Val Loss : %.5f, Error : %.5f" % (loss.item(), error.item()))
    net.train()
        

def train(net, optimizer, loaders, args):
    train_loader, val_loader, test_loader = loaders
    #eg = Rings() # EightGaussian()
    gauss = torch.distributions.normal.Normal(torch.tensor([0.]).cuda(), torch.tensor([1.]).cuda())
    #net = net.float()
    label = None
    for epoch in range(1, args.epoch+1):
        running_loss = 0.0
        for idx, (X, Y) in tqdm(enumerate(train_loader), total=len(train_loader)):
            #label = X[0]
            #print('Y is : ' + str(Y[0]))
            #break
            u = unif(size=(args.batch_size, args.dims))
            u = gauss.icdf(u)
            optimizer.zero_grad()
            X = net.f(X) #bn1(X)
            alpha, beta = net(u)
            loss = dual(U=u, Y_hat=(alpha, beta), Y=Y, X=X, eps=args.eps)
            loss.backward()
            optimizer.step()
            #for p in positive_params:
            #    p.data.copy_(torch.relu(p.data))
            running_loss += loss.item()
        #break
        #if epoch % (args.epoch//10) == 0:
        print('%.5f' %
            (running_loss/(idx+1)))
        validate(net, val_loader, args)

    test(net, args, name='imgs/trained.png', loader=test_loader, label=label)
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
    parser.add_argument('--quantile', default=0.5, type=float)
    parser.add_argument('--genTheor', action='store_true')
    parser.add_argument('--gaussian_support', action='store_true')
    parser.add_argument('--eps', default=0, type=float)
    parser.add_argument('--dataset', default='energy', type=str)
    parser.add_argument('--folder', type=str)
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--weights', default='', type=str)
    args = parser.parse_args()

    if args.dataset == 'energy':
        xdim = 28
        args.dims = 28

    elif args.dataset == 'stock':
        xdim = 3
        args.dims = 6

    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    torch.cuda.set_device('cuda:0')
    net = ConditionalConvexQuantile(xdim=xdim, 
                                    a_hid=128,
                                    a_layers=3,
                                    b_hid=128,
                                    b_layers=1,
                                    args=args)

    #net.apply(net.weights_init_uniform_rule)

    if len(args.weights) > 0:
        load(net, args.weights + '/net.pth')

    ds = [TimeSeriesDataset(dataset=args.dataset, device=device, split=x) for x in ['train', 'val', 'test']]
    loaders = [data.DataLoader(d, batch_size=args.batch_size, shuffle=True, drop_last=True) for d in ds]
    optimizer = optimizer(net, args)
    net.to(device)
    train(net, optimizer, loaders, args)
    #mnist
    #train(net, optimizer, loader, ds.y[:args.n].float().cuda(), args)

    if args.save_model:
        save(net, args.folder, 'net')

    if args.genTheor:
        Y = torch.from_numpy(ds.y)
        plotaxis(Y, name='imgs/theor')
        plot2d(Y, name='imgs/theor.png')

    print("Training completed!")
