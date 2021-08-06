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
import matplotlib.pyplot as plt
import seaborn as sns
import utils
from gen_data import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Synthetic(data.Dataset):
    def __init__(self, args,  n=1000):
        self.n = n
        #self.y = np.random.normal(loc=args.mean, scale=args.std, size=(self.n, 1))
        #self.y = np.random.normal(loc=args.mean, scale=args.std, size=(args.batch_size, args.dims))
        #self.y = np.random.exponential(scale=1.0, size=(args.batch_size, 1))
        #self.y = gaussian_mixture(means=[-3, 1, 8], stds=[0.5, 0.5, 0.5], p=[0.1, 0.6, 0.3], args=args)
        #self.y = np.random.multivariate_normal(mean=[2, 3], cov=np.array([[3,-2],[-2,5]]), size=(self.n))

        #gaussian checkerboard
        self.y = np.load('../data/synthetic/bary_ot_checkerboard_3.npy', allow_pickle=True).tolist()
        self.y = self.y['Y']

        #spiral
        self.y, _ = make_spiral(n_samples_per_class=self.n, n_classes=1,
            n_rotations=2.5, gap_between_spiral=0.1, noise=0.2,
                gap_between_start_point=0.1, equal_interval=True)

    def __len__(self):
        return len(self.y)#self.n

    def __getitem__(self, i):
        return torch.from_numpy(self.y[i])

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

class RNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=1, num_classes=2):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        #self.fc1 = nn.Linear(hidden_size, 1)
        self.fc2 = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        #out1 = self.fc1(out[:, 0, :])
        out = self.fc2(out[:, -1, :])
        #out = torch.cat([out1, out2], dim=-1)
        return out

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
    #cl = utils.cov(output.permute(-1,-2)) - utils.cov(target.permute(-1,-2))
    #cl = torch.norm(cl, p=1)
    return loss.mean() #+ 0.01*cl

def test(net, args, name):
    net.eval()
    tsr = None
    with torch.no_grad():
        for i in range(100): # get args.batch_size x 100 samples
            U = np.random.uniform(0, 1, size=(args.batch_size, args.dims, 1))
            U = torch.from_numpy(U).float().cuda()
            Y_hat = net(U).squeeze(-1).cuda()
            if tsr == None:
                tsr = Y_hat
            else:
                tsr = torch.cat([tsr, Y_hat], dim=0)
    print(tsr.shape)
    #histogram(tsr, name) # uncomment for 1d case
    plot2d(tsr, name='imgs/2d.png') # 2d contour plot
    plotaxis(tsr, name='imgs/train')

def train(net, optimizer, loader, args):
    for epoch in range(1, args.epoch+1):
        running_loss = 0.0
        for idx, batch in enumerate(loader):
            optimizer.zero_grad()
            loss = 0
            Y = batch.cuda()
            for j in range(args.m):
                #u = np.random.uniform(0, 1, size=(args.batch_size, 1))#U[:, j].unsqueeze(-1)
                #u = torch.from_numpy(u).float()
                u = torch.rand(size=(args.batch_size, args.dims, 1)).cuda()
                Y_hat = net(u)
                #print(u.shape, Y_hat.shape, Y.shape)
                loss += huber_quantile_loss(Y_hat, Y, u.squeeze(-1), reduce=True)
            loss /= args.m
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print('%.5f' %
        (running_loss/args.iters))
    test(net, args, name='imgs/trained.png')
    '''
    Y = np.random.multivariate_normal(mean=[2, 3], cov=np.array([[3,-2],[-2,5]]), size=(args.batch_size*100))
    Y = torch.from_numpy(Y)
    plotaxis(Y, name='imgs/theor')
    plot2d(Y, name='imgs/theor.png')
    '''

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
    parser.add_argument('--dims', default=2, type=int)
    parser.add_argument('--m', default=10, type=int)
    parser.add_argument('--n', default=10000, type=int)

    args = parser.parse_args()

    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    #trainloader, testloader = dataloader(args)
    #net = QNN(args)
    net = RNN().cuda()
    ds = Synthetic(args, n=args.n)
    loader = data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    #criterion = nn.CrossEntropyLoss()
    optimizer = optimizer(net, args)
    train(net, optimizer, loader, args)
    #test(net, criterion, testloader, args)
    print("Training completed!")
