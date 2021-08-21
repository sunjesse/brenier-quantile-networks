import torch
import torch.nn as nn
import torch.nn.functional as F
from ot_modules.icnn import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def dual(U, Y_hat, Y, X, eps=0):
    alpha, beta = Y_hat # alpha(U) + beta(U)^{T}X
    loss = torch.mean(alpha)
    Y = Y.permute(1, 0)
    X = X.permute(1, 0)
    BX = torch.mm(beta, X)
    UY = torch.mm(U, Y)
    # (U, Y), (U, X), beta.shape(bs, nclass), X.shape(bs, nclass)
    #print(BX.shape, UY.shape, alpha.shape)
    psi = UY - alpha - BX
    sup, _ = torch.max(psi, dim=0)
    #print(sup.shape)
    loss += torch.mean(sup)

    if eps == 0:
        return loss

    l = torch.exp((psi-sup)/eps)
    loss += eps*torch.mean(l)
    return loss

def dual_unconditioned(U, Y_hat, Y, eps=0):
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

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)

class MLPVAE(nn.Module):
    def __init__(self, args):
        super(MLPVAE, self).__init__()

        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc31 = nn.Linear(256, args.dims)
        self.fc32 = nn.Linear(256, args.dims)
        self.fc4 = nn.Linear(args.dims, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 784)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)

    def reparameterize(self, mu, logvar):
        return reparameterize(mu, logvar) if self.training else mu

    def decode(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return torch.sigmoid(self.fc6(h))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

class VAE(nn.Module):
    def __init__(self, image_size, channel_num, kernel_num, z_size):
        # configurations
        super().__init__()
        self.image_size = image_size
        self.channel_num = channel_num
        self.kernel_num = kernel_num
        self.z_size = z_size

        # encoder
        self.encoder = nn.Sequential(
            self._conv(channel_num, kernel_num // 4),
            self._conv(kernel_num // 4, kernel_num // 2),
            self._conv(kernel_num // 2, kernel_num),
        )

        # encoded feature's size and volume
        self.feature_size = image_size // 8
        self.feature_volume = kernel_num * (self.feature_size ** 2)

        # q
        self.q_mean = self._linear(self.feature_volume, z_size, relu=False)
        self.q_logvar = self._linear(self.feature_volume, z_size, relu=False)

        # projection
        self.project = self._linear(z_size, self.feature_volume, relu=False)

        # decoder
        self.decoder = nn.Sequential(
            self._deconv(kernel_num, kernel_num // 2),
            self._deconv(kernel_num // 2, kernel_num // 4),
            self._deconv(kernel_num // 4, channel_num, last=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        # encode x
        encoded = self.encoder(x)

        # sample latent code z from q given x.
        mean, logvar = self.q(encoded)
        z = self.z(mean, logvar)
        z_projected = self.project(z).view(
            -1, self.kernel_num,
            self.feature_size,
            self.feature_size,
        )

        # reconstruct x from z
        x_reconstructed = self.decoder(z_projected)
        # return the parameters of distribution of q given x and the
        # reconstructed image.
        return (mean, logvar), x_reconstructed, z

    # ==============
    # VAE components
    # ==============

    def q(self, encoded):
        unrolled = encoded.view(-1, self.feature_volume)
        return self.q_mean(unrolled), self.q_logvar(unrolled)

    def z(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        eps = (
            Variable(torch.randn(std.size())).cuda() if self._is_on_cuda else
            Variable(torch.randn(std.size()))
        )
        return eps.mul(std).add_(mean)

    def reconstruction_loss(self, x_reconstructed, x):
        return nn.BCELoss(size_average=False)(x_reconstructed, x) / x.size(0)

    def kl_divergence_loss(self, mean, logvar):
        return ((mean**2 + logvar.exp() - 1 - logvar) / 2).mean()
    # =====
    # Utils
    # =====

    def sample(self, size):
        z = Variable(
            torch.randn(size, self.z_size).cuda() if self._is_on_cuda() else
            torch.randn(size, self.z_size)
        )
        z_projected = self.project(z).view(
            -1, self.kernel_num,
            self.feature_size,
            self.feature_size,
        )
        return self.decoder(z_projected)

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    # ======
    # Layers
    # ======

    def _conv(self, channel_size, kernel_num):
        return nn.Sequential(
            nn.Conv2d(
                channel_size, kernel_num,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.BatchNorm2d(kernel_num),
            nn.ReLU(),
        )

    def _deconv(self, channel_num, kernel_num, last=False):
        if last:
            return nn.ConvTranspose2d(channel_num, kernel_num,
                                    kernel_size=4, stride=2, padding=1)
        return nn.Sequential(
            nn.ConvTranspose2d(
                channel_num, kernel_num,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.BatchNorm2d(kernel_num),
            nn.ReLU(),
        )

    def _linear(self, in_size, out_size, relu=True):
        return nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.ReLU(),
        ) if relu else nn.Linear(in_size, out_size)


class ConditionalConvexQuantile(nn.Module):
    def __init__(self, xdim, args, a_hid=512, a_layers=3, b_hid=512, b_layers=1):
        super(ConditionalConvexQuantile, self).__init__()
        self.xdim = xdim
        self.a_hid=a_hid
        self.a_layers=a_layers
        self.b_hid=b_hid
        self.b_layers=b_layers
        self.alpha = ICNN_LastInp_Quadratic(input_dim=args.dims,
                                 hidden_dim=self.a_hid,#1024,#512
                                 activation='celu',
                                 num_layer=self.a_layers)
        self.beta = ICNN_LastInp_Quadratic(input_dim=args.dims,
                                 hidden_dim=self.b_hid,
                                 activation='celu',
                                 num_layer=self.b_layers,
                                 out_dim=self.xdim)
        #self.fc_x = nn.Linear(self.xdim, self.xdim)

    def forward(self, z):
        # we want onehot for categorical and non-ordinal x.
        #x = self.to_onehot(x)
        alpha = self.alpha(z)
        beta = self.beta(z) #torch.bmm(self.beta(z).unsqueeze(1), self.fc_x(x).unsqueeze(-1))
        return alpha, beta
    
    def grad(self, u, x):
        x = self.to_onehot(x)
        u.requires_grad = True 
        phi = self.alpha(u).sum() + (torch.bmm(self.beta(u).unsqueeze(1), x.unsqueeze(-1)).squeeze(-1)).sum()
        d_phi = torch.autograd.grad(phi, u, create_graph=True)[0]
        return d_phi

    def invert(self, y):
        raise NotImplementedError
    
    def to_onehot(self, x):
        with torch.no_grad():
            onehot = torch.zeros((x.shape[0], self.xdim), device=device)
            onehot.scatter_(dim=-1, index=x.view(x.shape[0], 1), value=1)
            onehot -= 1/self.xdim
        #print(onehot)
        return onehot

