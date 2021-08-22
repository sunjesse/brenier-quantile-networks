import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=RuntimeWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
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
from models import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def loss_function(recon_x, x, mu, logvar):

    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + args.kl_scale * KLD

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

def optimizer(net, vae, args):
    assert args.optimizer.lower() in ["sgd", "adam"], "Invalid Optimizer"

    params = list(vae.parameters()) + list(net.parameters())
    if args.optimizer.lower() == "sgd":
	       return optim.SGD(params, lr=args.lr, momentum=args.beta1, nesterov=args.nesterov)
    elif args.optimizer.lower() == "adam":
	       return optim.Adam(params, lr=args.lr, betas=(args.beta1, args.beta2))

def unif(size, eps=1E-7):
    return torch.clamp(torch.rand(size).cuda(), min=eps, max=1-eps)

def test(net, args, name, loader, vae):
    net.eval()
    vae.eval()
    gauss = torch.distributions.normal.Normal(torch.tensor([0.]).cuda(), torch.tensor([1.]).cuda())
    U = unif(size=(100, args.dims))
    U = gauss.icdf(U)
    #U.requires_grad = True
    a = torch.arange(0, 10, device=device)
    X = a*torch.ones((10, 10), device=device).long()
    X = X.permute(1, 0).flatten()
    Y_hat = net.grad(U, X)#= net.forward(U, grad=True).sum()
    #f = net(U).sum()
    #Y_hat = torch.autograd.grad(f, U, create_graph=True)[0]
    print("max and min points generated: " + str(Y_hat.max()) + " " + str(Y_hat.min()))
    #z = torch.randn(100, 2, device=device)
    Y_hat = vae.decode(Y_hat)
    Y_hat = Y_hat.view(100, 28, 28).unsqueeze(1)
    utils.save_image(utils.make_grid(Y_hat, nrow=10),
        './mnist.png')
    return

positive_params = []

def train(net, optimizer, loader, vae, args):
    k = args.k
    gauss = torch.distributions.normal.Normal(torch.tensor([0.]).cuda(), torch.tensor([1.]).cuda())
    for epoch in range(1, args.epoch+1):
        running_loss = 0.0
        dual_loss = 0.0
        for idx, (Y, label) in enumerate(loader):
            Y = Y.cuda()
            label = label.cuda()
            u = unif(size=(args.batch_size, args.dims))
            u = gauss.icdf(u)
            optimizer.zero_grad()
            alpha, beta = net(u)
            #Y_hat = net(u)
            X = net.to_onehot(label)
            Y_recon, mu, logvar, z = vae(Y)
            #if epoch <= args.epoch // 2:
            loss = loss_function(Y_recon, Y, mu, logvar)#vae.reconstruction_loss(Y_recon, Y)
            l1 = loss.item()
            #else:
            q_loss = dual(U=u, Y_hat=(alpha, beta), Y=mu.detach(), X=X, eps=args.eps)
            if q_loss.item() > 0:
                loss += q_loss#dual(U=u, Y_hat=(alpha, beta), Y=z.detach(), X=X, eps=args.eps)
            l2 = loss.item()
            dual_loss += l2 - l1

            #loss += dual_unconditioned(U=u, Y_hat=Y_hat, Y=mu.detach(), eps=args.eps)
            loss.backward()
            optimizer.step()
            for p in positive_params:
            	p.data.copy_(torch.relu(p.data))
            running_loss += loss.item()

        print('Epoch %d : %.5f %.5f' %
            (epoch, running_loss/len(loader.dataset), dual_loss/len(loader.dataset)))

    test(net, args, name='imgs/trained.png', loader=loader, vae=vae)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # optimization related arguments
    parser.add_argument('--batch_size', default=128, type=int,
                        help='input batch size')
    parser.add_argument('--epoch', default=25, type=int,
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
    parser.add_argument('--dims', default=3, type=int)
    parser.add_argument('--m', default=10, type=int)
    parser.add_argument('--n', default=5000, type=int)
    parser.add_argument('--k', default=100, type=int)
    parser.add_argument('--genTheor', action='store_true')
    parser.add_argument('--gaussian_support', action='store_true')
    parser.add_argument('--eps', default=0, type=float)
    parser.add_argument('--kl_scale', default=1., type=float)
    args = parser.parse_args()

    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    torch.cuda.set_device('cuda:0')
    net = ConditionalConvexQuantile(xdim=10, 
                                    args=args,
                                    a_hid=128, 
                                    a_layers=2,
                                    b_hid=128,
                                    b_layers=1)
    '''
    net = ICNN_LastInp_Quadratic(input_dim=args.dims,
                        hidden_dim=512,
                        activation='celu',
                        num_layer=3)
    '''
    '''
    vae = VAE(image_size=32,
            channel_num=1,
            kernel_num=128,
            z_size=args.dims)
    '''
    vae = MLPVAE(args=args)

    for p in list(net.parameters()):
        if hasattr(p, 'be_positive'):
            positive_params.append(p)
        p.data = torch.from_numpy(truncated_normal(p.shape, threshold=1./np.sqrt(p.shape[1] if len(p.shape)>1 else p.shape[0]))).float()

    transform=transforms.Compose([transforms.ToTensor()])
    ds = datasets.MNIST('../data', train=True, download=True,transform=transform)
    loader = data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    optimizer = optimizer(net, vae, args)
    net.cuda()
    vae.cuda()
    train(net, optimizer, loader, vae, args)
    #mnist
    #train(net, optimizer, loader, ds.y[:args.n].float().cuda(), args)

    if args.genTheor:
        Y = torch.from_numpy(ds.y)
        plotaxis(Y, name='imgs/theor')
        plot2d(Y, name='imgs/theor.png')

    print("Training completed!")
