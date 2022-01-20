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
from utils import *
from dataloader import *
from models import *
from gen_data import *
from tqdm import tqdm
from utils2 import *
from sklearn.preprocessing import MinMaxScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N = 30 
wd = 100 
Qs = [0.025, 0.25, 0.5, 0.75, 0.975]
scaler = MinMaxScaler()

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

def l1_quantile_loss(output, target, tau, reduce=True):
    u = target - output
    loss = (tau - (u.detach() <= 0).float()).mul_(u)
    return loss.mean() if reduce else loss


def plot_timeseries(X, preds, i, Ytrue):
    h, w = 1, 1
    feats = [0]
    assert len(feats) == w

    X = X.detach().cpu().numpy()
    Ytrue = Ytrue.detach().cpu().numpy()

    X = scaler.inverse_transform(X[:, :, 0])[:, :, None]
    Ytrue = scaler.inverse_transform(Ytrue[:, :, 0])[:, :, None]

    print(Ytrue.shape)
    f, axs = plt.subplots(h, w, figsize=(14, 14), sharex=True, sharey=True)
    for c, i in enumerate(list(np.random.randint(0, X.shape[0], size=(h)))):
        for pltnum, k in enumerate(feats):
            xs = X[i, :, k]
            yt = Ytrue[i, :, k]
            t = np.arange(0, xs.shape[0], 1)
            axs.plot(t, xs)
            prev = None
            for idx, y in enumerate(preds):
                ys = y.detach().cpu().numpy()
                ys = scaler.inverse_transform(ys[:, :, 0])[:, :, None]
                ys = ys[i, :, k]
                p_t, p_x = t[-1], xs[-1]
                ys = [p_x]+[ys[t] for t in range(wd)]
                ts = [p_t+j for j in range(wd+1)]
                if prev != None:
                    if idx in [0, 4]:
                        l = '95% confidence interval'
                    elif idx in [3]:
                        l = '50% confidence interval'
                    else:
                        l = None
                    axs.fill_between(ts, prev, ys, alpha=0.75*(1-abs(idx-2.5)/3), color='g', edgecolor=None, label=l)
                if idx == 2:
                    axs.plot(ts, ys, alpha=1, label=str(Qs[idx])+" quantile level" if c+pltnum == 0 else None)#color='r') #[p_x]+[ys[t] for t in range(wd)], label=str((idx+1)/10.))
                prev = ys
            axs.plot([p_t+j for j in range(0, 101)], np.concatenate([[xs[-1]], yt], axis=0), color='tab:blue', label="target" if c+pltnum == 0 else None)
            #axs[i, pltnum].set_axis_off()
    f.legend()
    #f.tight_layout()
    f.savefig("./ts.png")
    print('saved fig!')

def getXY(loader):
    X, Y = [], []
    i = 0
    for (x, y) in loader:
        if i < 210 or i % 30 != 0:
            i+=1
            continue
        X.append(x)
        Y.append(y)
        i += 1
        print(i)
        if i == 210 + N*5+1:
            break
        #return x.float(), y.float()
    return torch.cat(X, 1).float(), torch.cat(Y, 1).float()

def test(net, args, name, loader):
    net.eval()
    X, Y = getXY(loader)
    print(X.shape, Y.shape)
    #Y = Y.cuda()
    gauss = torch.distributions.normal.Normal(torch.tensor([0.]).cuda(), torch.tensor([1.]).cuda())
    #X = X.cuda()   
    #X, Ytrue = X[:, :12, :], X[:, 12:, :]
    preds = []
    Xt = X[:, :30]
    '''
    for q in [0.5]:
        U = torch.ones(X.shape[0], args.dims)*q
        U = U.cuda()
        U = gauss.icdf(U)
        for ti in range(wd):
            Y_hat, Yv = net.grad(U, Xt[:, ti:], onehot=False)
            #Xt = torch.cat([Xt, Y_hat.unsqueeze(1).detach()], axis=1)
            Xt = torch.cat([Xt, Y_hat[:, None].detach()], axis=1)
            print(Xt.shape)
            #print(Xt.shape, Xt[:, ti+1:].shape)
    '''

    for q in Qs: #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]: #np.linspace(0.025, 0.975, N):
        U = torch.ones(X.shape[0], args.dims)*q
        U = U.cuda()
        U = gauss.icdf(U)
        tmp = []
        Xc = Xt 
        for ti in range(100):
            #print(Xt[0, ti, 27])
            #_, Yv = net.grad(U, Xt, onehot=False)
            #Xt = torch.cat([Xt[:, 1:], Yv[:, None]], axis=1)
            Y_hat, _ = net.grad(U, Xc, onehot=False)
            #Xc = torch.cat([Xc[:, 1:], Y_hat[:, None].detach()], 1)
            Xc = torch.cat([Xc[:, 1:], X[:, 30+ti, None]], 1)
            tmp.append(Y_hat.unsqueeze(1))
        preds.append(torch.cat(tmp, axis=1))
    #preds = preds[:4] + [Xt[:, -3:]] + preds[4:]
    plot_timeseries(X[:, :30], preds, 0, X[:, 30:130])
    '''
    epsilon = torch.abs(Y_hat - Y)
    ql = l1_quantile_loss(Y_hat, Y, U)
    print('max : ' + str(epsilon.max().item()))
    print('mae : ' + str(epsilon.mean().item()))
    print('ql' + str(args.quantile*100) + ': ' + str(ql.item()))
    Y_hat, Y, U = Y_hat.detach().cpu().numpy(), Y.detach().cpu().numpy(), U.detach().cpu().numpy()
    print("rmse: " + str(rmse(Y, Y_hat)))
    print("smape: " + str(smape(Y, Y_hat)))
    U = torch.ones(X.shape[0], args.dims)*0.9
    U = U.cuda()
    Y_hat = net.grad(U, X, onehot=False)
    ql90 = l1_quantile_loss(Y_hat, torch.from_numpy(Y).cuda(), U.cuda())
    print("ql90: " + str(ql90.item()))
    '''
def validate(net, loader, args):
    net.eval()
    gauss = torch.distributions.normal.Normal(torch.tensor([0.]).cuda(), torch.tensor([1.]).cuda())
    X, Y = loader.dataset.getXY()
    #u = unif(size=(X.shape[0], args.dims))
    u = torch.ones(size=(X.shape[0], args.dims))*args.quantile
    u = u.cuda()
    u = gauss.icdf(u)
    X = net.f(X)#bn1(X)
    alpha, beta = net(u)
    loss = dual(U=u, Y_hat=(alpha, beta), Y=Y, X=X, eps=args.eps)
    Y_hat = net.grad(u, X, onehot=False)
    #error = F.mse_loss(Y_hat, Y, reduction='mean')
    error = torch.abs(Y_hat - Y).mean()#torch.sum((Y_hat - Y)**2/(Y_hat.shape[0]*Y_hat.shape[1]))
    print("Val Loss : %.5f, Error : %.5f" % (loss.item(), error.item()))
    net.train()
        
positive_params = []

def train(net, optimizer, loaders, args):
    train_loader, val_loader, test_loader = loaders
    #eg = Rings() # EightGaussian()
    gauss = torch.distributions.normal.Normal(torch.tensor([0.]).cuda(), torch.tensor([1.]).cuda())
    #net = net.float()
    for epoch in range(1, args.epoch+1):
        running_loss = 0.0
        for idx, (X, Y) in enumerate(train_loader):#for idx, (X, Y) in tqdm(enumerate(train_loader), total=len(train_loader)):
            #X, Y = X[:, :N-wd], X[:, N-wd]
            #X=X.float().cuda()
            #Y=Y.float().cuda()
            u = unif(size=(args.batch_size, args.dims))
            u = gauss.icdf(u)
            optimizer.zero_grad()
            y_norm, yv = net.f(X) #bn1(X)
            alpha, beta = net(u)
            loss = dual(U=u, Y_hat=(alpha, beta), Y=Y, X=y_norm, eps=args.eps)
            #loss = F.mse_loss(yv, Y)
            loss.backward()
            optimizer.step()
            for p in positive_params:
                p.data.copy_(torch.relu(p.data))
            running_loss += loss.item()
        if (epoch % (args.epoch//10)) == 0:
            print('%.5f' %
                (running_loss/(idx+1)))
        #validate(net, val_loader, args)

    test(net, args, name='imgs/trained.png', loader=test_loader)

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
    parser.add_argument('--eps', default=0, type=float)
    parser.add_argument('--quantile', default=0.5, type=float)
    parser.add_argument('--dataset', default='energy', type=str)
    parser.add_argument('--folder', type=str)
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--weights', default='', type=str)
    args = parser.parse_args()

    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    if args.dataset == 'energy':
        xdim = 1
        args.dims = 1

    elif args.dataset == 'stock':
        xdim = 3 
        args.dims = 6

    net = ConditionalConvexQuantile(xdim=xdim, 
                                    a_hid=128,
                                    a_layers=3,
                                    b_hid=128,
                                    b_layers=3,
                                    args=args)

    data = load_data('./data/energydata_complete.csv')[['Appliances']]
    data = data.copy()

    # Plot Time-Series Data #
    if False:
        plot_full(args.plots_path, data, args.feature)

    data['Appliances'] = scaler.fit_transform(data)

    # Split the Dataset #
    copied_data = data.copy().values

    #del data
    #data = None

    if False:
        X, y = split_sequence_multi_step(copied_data, N, 1)
        step = 'MultiStep'
    else:
        X, y = split_sequence_uni_step(copied_data, N)
        step = 'SingleStep'

    train_loader, val_loader, test_loader = data_loader(X, y, 0.8, 0.5, args.batch_size)
    '''
    net = BiRNN(input_size=1,
                   hidden_size=64,#args.dims*4,
                   num_layers=3,
                   xdim=1)
    '''
    for p in list(net.parameters()):
        if hasattr(p, 'be_positive'):
            positive_params.append(p)
        p.data = torch.from_numpy(truncated_normal(p.shape, threshold=1./np.sqrt(p.shape[1] if len(p.shape)>1 else p.shape[0]))).float()

    if len(args.weights) > 0:
        load(net, args.weights + '/net.pth')

    ds = [TimeSeriesDataset(dataset=args.dataset, device=device, split=x) for x in ['train', 'val', 'train']]
    #loaders = [data.DataLoader(d, batch_size=args.batch_size, shuffle=False, drop_last=True) for d in ds]
    optimizer = optimizer(net, args)
    net.to(device)
    train(net, optimizer, [train_loader, val_loader, test_loader], args)
    
    if args.save_model:
        save(net, args.folder, 'net')

    print("Training completed!")
