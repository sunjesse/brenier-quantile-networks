import numpy as np
import torch
from scipy.stats import truncnorm
import os

def gen_random_projection(M=100, d=2): # generates M samples of dimension d on a d-sphere uniformly.
    W = np.random.multivariate_normal(mean=np.zeros(d), cov=np.eye(d), size=(M))
    z = np.sqrt(np.sum(W*W, axis=1))
    z = np.expand_dims(z, axis=1)
    z = np.concatenate([z, z], axis=1)
    W /= z + 1E-5
    return torch.from_numpy(W).float()

def cov(X):
    D = X.shape[-1]
    mean = torch.mean(X, dim=-1).unsqueeze(-1)
    X = X - mean
    return 1/(D-1) * X @ X.transpose(-1, -2)

def linear(M=100, d=2):
    theta=torch.randn((M,d))
    theta=torch.stack([th/torch.sqrt((th**2).sum()) for th in theta])
    return theta

def truncated_normal(size, threshold=1):
    values = truncnorm.rvs(-threshold, threshold, size=size)
    return values

def save(model, folder, name):
    direct = os.path.join(os.getcwd(), 'ckpt', folder)
    name = name + '.pth'
    if os.path.exists(direct) == False:
        os.mkdir(direct)
    torch.save(model.state_dict(),
                '{}/{}'.format(direct, name))
    print('Saved weights to ' + direct + '!')

def load(model, weights):
    direct = os.path.join(os.getcwd(), 'ckpt', weights)
    model.load_state_dict(
        torch.load(direct, map_location=lambda storage, loc: storage), strict=False)
    print('Loaded weights: ' + str(weights) + '!')

def msis(target, lower_quantile, upper_quantile, seasonal_error, alpha):
    r"""
    .. math::

        msis = mean(U - L + 2/alpha * (L-Y) * I[Y<L] + 2/alpha * (Y-U) * I[Y>U]) / seasonal\_error

    See [SSA20]_ for more details.
    """
    numerator = np.mean(
        upper_quantile
        - lower_quantile
        + 2.0 / alpha * (lower_quantile - target) * (target < lower_quantile)
        + 2.0 / alpha * (target - upper_quantile) * (target > upper_quantile)
    )

    return numerator / seasonal_error

def smape(target, forecast):
    r"""
    .. math::

        smape = 2 * mean(|Y - \hat{Y}| / (|Y| + |\hat{Y}|))

    See [HA21]_ for more details.
    """
    return 2 * np.mean(
        np.abs(target - forecast) / (np.abs(target) + np.abs(forecast))
    )

def rmse(target, forecast):
    if target.ndim == 2:
        N = target.shape[0]*target.shape[1]
    else:
        N = target.shape[0]
    mse = np.sum((target-forecast)**2)/N
    return np.sqrt(mse)

def quantile_loss(target: np.ndarray, forecast: np.ndarray, q: float) -> float:
    r"""
    .. math::
        quantile\_loss = 2 * sum(|(Y - \hat{Y}) * (Y <= \hat{Y}) - q|)
    """
    return 2 * np.mean(np.abs((forecast - target) * ((target <= forecast) - q)))
