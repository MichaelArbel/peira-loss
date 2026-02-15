import torch
from .linalg_utils import PDInverse, add_ridge, solve_with_cholesky



def l2_penalty(param_list):
    pen = 0.
    for param in param_list:
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


def exact_linear_probe_loss(X: torch.Tensor, Y: torch.Tensor, reg: float, symmetric=False):
    with torch.no_grad():
        B = X.shape[0]
        inter_cov = torch.einsum("mk,ml->kl", Y, X) / B
        intra_cov = torch.einsum("mk,ml->kl", X, X) / B
        pred = solve_with_cholesky(inter_cov, add_ridge(intra_cov, reg))

    loss = linear_probe_loss(X, Y, pred.detach(), reg, symmetric=symmetric)
    return loss



def eckhart_young_loss(X: torch.Tensor, Y: torch.Tensor):
    B = X.shape[0]
    inter_cov = torch.einsum("mk,mk->", X, Y) / B
    cov_u = torch.einsum("mk,ml->kl", X, X) / B
    cov_v = torch.einsum("mk,ml->kl", Y, Y) / B
    intra_cov = torch.sum((cov_u + cov_v) ** 2)

    return -4.0 * inter_cov + intra_cov


def eckhart_young_sto_loss(X: torch.Tensor, Y: torch.Tensor):
    B = X.shape[0]
    inter_cov = torch.einsum("mk,mk->", X, Y) / B
    X1, X2 = torch.tensor_split(X, 2, dim=0)
    Y1, Y2 = torch.tensor_split(Y, 2, dim=0)

    cov_u1 = torch.einsum("mk,ml->kl", X1, X1) / X1.shape[0]
    cov_v1 = torch.einsum("mk,ml->kl", Y1, Y1) / Y1.shape[0]

    cov_u2 = torch.einsum("mk,ml->kl", X2, X2) / X2.shape[0]
    cov_v2 = torch.einsum("mk,ml->kl", Y2, Y2) / Y2.shape[0]


    intra_cov = torch.sum((cov_u1 + cov_v1) * (cov_u2 + cov_v2))

    return -4.0 * inter_cov + intra_cov







