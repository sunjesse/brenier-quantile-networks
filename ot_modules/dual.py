import torch

def dual(U, Y_hat, Y, eps=1):
    loss = torch.mean(Y_hat)
    U = U.unsqueeze(1)
    Y = Y.unsqueeze(-1)
    psi = torch.bmm(U, Y).squeeze(-1) - Y_hat
    #loss += torch.max(psi)
    #return loss
    l = torch.exp(psi/eps)
    loss += eps*torch.mean(l)
    return loss
