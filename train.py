import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import argparse
import numpy as np
import scipy
from scipy.stats import norm
#our libs
from lib import radam
import matplotlib.pyplot as plt
import seaborn as sns

class QNN(nn.Module):
    def __init__(self, args):
        super(QNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(args.dims, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, args.dims))

    def forward(self, x):
        return self.net(x)

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

def gaussian_mixture(means, stds, p, args):
    assert np.sum(p) == 1
    k = len(p)
    ranges = [0.]
    for i in range(k):
        ranges.append(ranges[i] + p[i])
    mix = np.zeros((args.batch_size, 1))
    idx = np.random.uniform(0, 1, size=(args.batch_size, 1))
    for i in range(k):
        g = np.random.normal(loc=means[i], scale=stds[i], size=(args.batch_size, 1))
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
 
def criterion(pred, label, quantile):
    loss = (label-pred)*(quantile - (label-pred < 0).float())
    return torch.mean(loss)

def huber_quantile_loss(output, target, tau, k=0.02, reduce=True):
    u = target - output
    loss = (tau - (u.detach() <= 0).float()).mul_(u.detach().abs().clamp(max=k).div_(k)).mul_(u)
    return loss.mean() if reduce else loss

def test(net, args, name):
    net.eval()
    tsr = None
    with torch.no_grad():
        for i in range(100): # get args.batch_size x 100 samples
            U = np.random.uniform(0, 1, size=(args.batch_size, args.dims))
            U = torch.from_numpy(U).float()
            Y_hat = net(U)
            if tsr == None:
                tsr = Y_hat
            else:
                tsr = torch.cat([tsr, Y_hat], dim=0)
    #histogram(tsr, name) # uncomment for 1d case
    plot2d(tsr, name='2d.png') # 2d contour plot

def train(net, optimizer, args):
    '''
    Y = np.random.normal(loc=0.0, scale=1.0, size=(args.batch_size, 1))
    Y = torch.from_numpy(Y)
    U = np.random.uniform(0, 1, size=(args.batch_size, 1))
    U = torch.from_numpy(U).float()
    '''
    #test(net, args, name='untrained.png')
    for epoch in range(1, args.epoch+1):
        running_loss = 0.0
        for i in range(args.iters):
            U = np.random.uniform(0, 1, size=(args.batch_size, args.dims))
            #Y = np.random.normal(loc=args.mean, scale=args.std, size=(args.batch_size, args.dims))
            #Y = np.random.exponential(scale=1.0, size=(args.batch_size, 1))
            #Y = gaussian_mixture(means=[-3, 1, 8], stds=[0.5, 0.5, 0.5], p=[0.1, 0.6, 0.3], args=args)
            cov = np.array([[3, -2], [-2, 5]])
            Y = np.random.multivariate_normal(mean=[2, 3], cov=cov)
            U, Y = torch.from_numpy(U).float(), torch.from_numpy(Y)
            optimizer.zero_grad()
            Y_hat = net(U)
            loss = huber_quantile_loss(Y_hat, Y, U, reduce=False)
            loss = loss.mean()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('%.5f' %
			(running_loss/args.iters))
    test(net, args, name='trained.png')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # optimization related arguments
    parser.add_argument('--batch_size', default=128, type=int,
                        help='input batch size')
    parser.add_argument('--epoch', default=50, type=int,
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

    args = parser.parse_args()

    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    #trainloader, testloader = dataloader(args)
    net = QNN(args)
    #net = torch.nn.RNN(input_size=1, hidden_size=32, num_layers=3, nonlinearity='relu')
    #criterion = nn.CrossEntropyLoss()
    optimizer = optimizer(net, args)
    train(net, optimizer, args)
    #test(net, criterion, testloader, args)
    print("Training completed!")
