import torch
import torch.nn as nn
import torch.nn.functional as F
from ot_modules.icnn import *
from supp.distribution_output import *
from supp.piecewise_linear import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
attributes = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']


def dual(U, Y_hat, Y, X, eps=0):
    alpha, beta = Y_hat # alpha(U) + beta(U)^{T}X
    Y = Y.permute(1, 0)
    X = X.permute(1, 0)
    BX = torch.mm(beta, X)
    loss = torch.mean(alpha) #+ BX)
    UY = torch.mm(U, Y)
    # (U, Y), (U, X), beta.shape(bs, nclass), X.shape(bs, nclass)
    #print(BX.shape, UY.shape, alpha.shape)
    psi = UY - alpha - BX
    sup, _ = torch.max(psi, dim=0)
    #print(sup.shape)
    #print(UY.min(), UY.max(), sup.mean())
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

def generate_x():
    x = torch.zeros(40)
    with open('./description.txt') as f:
        for line in f:
            i = attributes.index(line[:-1])
            x[i] = 1
    return x

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)

class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, xdim, bn_last=True):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bn_last = bn_last
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, xdim)
        self.norm = nn.BatchNorm1d(xdim, momentum=1.0, affine=False)
    
    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) # 2 for bidirection 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        if self.bn_last:
            return self.norm(out), out
        return out

class ShallowRegressionLSTM(nn.Module):
    def __init__(self, num_sensors, hidden_units):
        super().__init__()
        self.num_sensors = num_sensors  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = 1

        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )

        self.linear = nn.Linear(in_features=self.hidden_units, out_features=num_sensors)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).to(device).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).to(device).requires_grad_()

        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(hn[0])  # First dim of Hn is num_layers, which is set to 1 above.
        print(out.shape)
        return None, out

class Spline(nn.Module):
    def __init__(self, args):
        super(Spline, self).__init__()
        self.f = BiRNN(input_size=1,
                        hidden_size=4,
                        num_layers=2,
                        xdim=50)
        self.d_out = PiecewiseLinearOutput(num_pieces=50)
        self.args_proj = self.d_out.get_args_proj(in_features=50)

    def forward(self, x, y=None, u=None):
        h = self.f(x)
        gamma, slopes, knot_spacings = self.args_proj(h)
        distr = PiecewiseLinear(gamma=gamma, slopes=slopes, knot_spacings=knot_spacings)
        if y != None:
            return distr.crps(y)
        return distr.sample()

class QuantileLayer(nn.Module):
    """Define quantile embedding layer, i.e. phi in the IQN paper (arXiv: 1806.06923)."""

    def __init__(self, num_output):
        super(QuantileLayer, self).__init__()
        self.n_cos_embedding = 64
        self.num_output = num_output
        self.output_layer = nn.Sequential(
            nn.Linear(self.n_cos_embedding, self.n_cos_embedding),
            nn.PReLU(),
            nn.Linear(self.n_cos_embedding, num_output),
        )

    def forward(self, tau):
        cos_embedded_tau = self.cos_embed(tau)
        final_output = self.output_layer(cos_embedded_tau)
        return final_output

    def cos_embed(self, tau):
        integers = torch.repeat_interleave(
            torch.arange(0, self.n_cos_embedding).unsqueeze(dim=0),
            repeats=tau.shape[-1],
            dim=0,
        ).to(tau.device)
        return torch.cos(math.pi * tau.unsqueeze(dim=-1) * integers)

class IQN(nn.Module):
    def __init__(self, args):
        super(IQN, self).__init__()
        self.f = BiRNN(input_size=args.dims,
                        hidden_size=args.dims*4,
                        num_layers=2,
                        xdim=50)
        self.phi = QuantileLayer(num_output=50)
        self.output_layer = nn.Sequential(nn.Linear(50, 50), 
                        nn.Softplus(),
                        nn.Linear(50, args.dims))
	
    def forward(self, tau, x):
        h = self.f(x)
        embedded_tau = self.phi(tau).squeeze(1)
        new_input_data = h * (torch.ones_like(embedded_tau) + embedded_tau)
        return self.output_layer(new_input_data)


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

        self.feat_last = hidden_dims[-1] 
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

    def decode(self, z, train=True):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, self.feat_last, 2, 2)
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

        '''

        alpha = []
        alpha.append(nn.Sequential(nn.Linear(args.dims, self.a_hid),
                                   #nn.BatchNorm1d(self.a_hid),
                                   nn.CELU(inplace=True)))

        for i in range(2, self.a_layers+1):
            alpha.append(nn.Sequential(nn.Linear(self.a_hid, self.a_hid),
                                       #nn.BatchNorm1d(self.a_hid),
                                       nn.CELU(inplace=True)))

        alpha.append(nn.Sequential(nn.Linear(self.a_hid, 1)))
        self.alpha = nn.Sequential(*alpha)
        self.beta = None
        if self.xdim > 0:
            beta = []
            beta.append(nn.Sequential(nn.Linear(args.dims, self.b_hid),
                                      #nn.BatchNorm1d(self.b_hid),
                                      nn.CELU(inplace=True)))

            for i in range(2, self.b_layers+1):
                beta.append(nn.Sequential(nn.Linear(self.b_hid, self.b_hid),
                                          #nn.BatchNorm1d(self.b_hid),
                                          nn.CELU(inplace=True)))

            beta.append(nn.Sequential(nn.Linear(self.b_hid, self.xdim)))
            self.beta = nn.Sequential(*beta)
            # BiRNN
        '''
        self.f = BiRNN(input_size=args.dims,
                       hidden_size=512,#args.dims*4,
                       num_layers=1,
                       xdim=self.xdim)
        #self.f = ShallowRegressionLSTM(1, 128)
        # MLP

        #self.bn1 = nn.BatchNorm1d(self.xdim, momentum=1.0, affine=False)

        #self.f = nn.BatchNorm1d(self.xdim, affine=False)

    def forward(self, z, x=None):
        # we want onehot for categorical and non-ordinal x.
        if self.xdim == 0:
            return self.alpha(z)
        alpha = self.alpha(z)
        beta = self.beta(z) #torch.bmm(self.beta(z).unsqueeze(1), self.fc_x(x).unsqueeze(-1))
        #quad = (z.view(z.size(0), -1) ** 2).sum(1, keepdim=True) / 2
        return alpha, beta #, self.fc_x(x)
    
    def grad(self, u, x=None, onehot=True):
        if onehot and self.xdim > 0:
            x = self.to_onehot(x)
        elif x != None:
            x, xv = self.f(x)#self.bn1(x)
        u.requires_grad = True 
        phi = self.alpha(u).sum()
        if self.xdim != 0 and x != None:
            phi += (torch.bmm(self.beta(u).unsqueeze(1), x.unsqueeze(-1)).squeeze(-1)).sum()
        d_phi = torch.autograd.grad(phi, u, create_graph=True)[0]
        return d_phi, xv

    def grad_multi(self, u, x):
        if x == None:
            x = generate_x()
        x_s = x.shape[-1]
        for i in range(40):
            if x[i] == 1:
                print(attributes[i], end=',')
        x = x.expand(1, x_s)
        x = x.repeat(u.shape[0], 1).float().cuda()
        x = self.f(x)
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
            #onehot -= 1/self.xdim
            #onehot = self.bn1(onehot)
        onehot = self.f(onehot)
        return onehot

    def weights_init_uniform_rule(self, m):
        classname = m.__class__.__name__
        # for every Linear layer in a model..
        if classname.find('Linear') != -1:
            # get the number of the inputs
            n = m.in_features
            y = 1.0/np.sqrt(n)
            m.weight.data.uniform_(-y, y)
            m.bias.data.fill_(0)
