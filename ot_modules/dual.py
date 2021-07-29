import torch

def dual(U, Y_hat, Y, eps=1):
    loss = torch.mean(Y_hat)
    U = U.unsqueeze(1)
    Y = Y.unsqueeze(-1)
    psi = torch.empty(Y_hat.shape)
    for i in range(Y_hat.shape[-1]):
        mm = torch.bmm(U[:, :, :, i], Y).squeeze(-1)
        psi[:, :, i] = mm - Y_hat[:, :, i]
    psi = psi.permute(2, 0, 1)
    sup, _ = torch.max(psi, dim=0)
    loss += torch.mean(sup)
    l = torch.exp(psi-sup/eps)
    loss += eps*torch.mean(l)
    return loss

def _dual(U, Y_hat, Y, eps=1): #dual w/o calculating the sup in psi and minimizing psi.
    loss = torch.mean(Y_hat)
    U = U.unsqueeze(1)
    Y = Y.unsqueeze(-1)
    psi = torch.bmm(U, Y).squeeze(-1) - Y_hat
    #loss += torch.max(psi)
    #return loss
    l = torch.exp(psi/eps)
    loss += eps*torch.mean(l)
    return loss
