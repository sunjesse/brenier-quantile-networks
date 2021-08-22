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
    def __init__(self,
                 in_channels,
                 latent_dim,
                 hidden_dims = None,
                 **kwargs):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Sigmoid())#Tanh())

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
       	result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 256, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

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

