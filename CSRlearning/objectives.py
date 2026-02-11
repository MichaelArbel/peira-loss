import torch
from linalg_utils import PDInverse, add_ridge



def l2_penalty(model):
    pen = 0.
    for param in model.parameters():
        pen += torch.sum(param ** 2)
    return pen

def semigrad_ssl_loss(X: torch.Tensor, Y: torch.Tensor, P: torch.Tensor, reg: float) -> torch.Tensor:
    pred = torch.einsum("rl,bl->br", P, X)
    pred_loss = 0.5 * torch.mean(torch.sum((pred - Y) ** 2, dim=1))
    reg_loss= 0.5 * reg * P.pow(2).sum()
    return pred_loss + reg_loss


def linear_probe_loss(X: torch.Tensor, Y: torch.Tensor, P: torch.Tensor, reg: float, symmetric=True) -> torch.Tensor:
    data_term = 0.5 * torch.mean(torch.sum((X @ P.T - Y) ** 2, dim=1))
    if symmetric:
        data_term = data_term + 0.5 * torch.mean(torch.sum((Y @ P.T - X) ** 2, dim=1))
    reg_term = 0.5 * reg * P.pow(2).sum()
    return data_term + reg_term


def adjoint_pred_loss(X: torch.Tensor, Y: torch.Tensor, P: torch.Tensor, reg: float) -> torch.Tensor:
    data_term = 0.5 * torch.mean(torch.sum((X @ P.T) ** 2, dim=1))
    data_term = data_term + 0.5 * torch.mean(torch.sum((Y @ P.T) ** 2, dim=1))
    data_term = data_term - torch.trace(P)
    reg_term = 0.5 * reg * P.pow(2).sum()
    loss = data_term + reg_term
    return loss


def cse_auxiliary_loss(X: torch.Tensor, Y: torch.Tensor, P: torch.Tensor, Q: torch.Tensor, reg: float) -> torch.Tensor:
    loss = 0.5 * reg * (torch.mean(torch.sum(X ** 2, dim=1)) + torch.mean(torch.sum(Y ** 2, dim=1)))
    loss = loss + 0.5 * torch.mean(torch.sum((X @ P.T - Y) * (X @ Q.T), dim=1))
    loss = loss + 0.5 * torch.mean(torch.sum((Y @ P.T - X) * (Y @ Q.T), dim=1))

    return loss


def cse_loss(X: torch.Tensor, Y: torch.Tensor, P: torch.Tensor, reg: float):
    reg_loss = 0.5 *reg * (
                torch.mean(torch.sum(X ** 2, dim=1)) + torch.mean(torch.sum(Y ** 2, dim=1))
            )

    if P is None:
        B = X.shape[0]
        inter_cov = torch.einsum("mk,ml->kl", Y, X) / B
        cov_u = torch.einsum("mk,ml->kl", X, X) / B
        cov_v = torch.einsum("mk,ml->kl", Y, Y) / B
        inv_cov = PDInverse.apply(add_ridge(cov_v+cov_u,reg))
        loss = -0.5 * torch.sum((inter_cov + inter_cov.t()) * inv_cov)
        return loss  + reg_loss
    else:
        return - 0.5 * torch.trace(P) + reg_loss




def eckhart_young_loss(X: torch.Tensor, Y: torch.Tensor):
    B = X.shape[0]
    inter_cov = torch.einsum("mk,mk->", X, Y) / B
    cov_u = torch.einsum("mk,ml->kl", X, X) / B
    cov_v = torch.einsum("mk,ml->kl", Y, Y) / B
    intra_cov = torch.sum((cov_u + cov_v) ** 2)

    return -4.0 * inter_cov + intra_cov






