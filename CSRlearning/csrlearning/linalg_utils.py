import math
import torch
import torch.nn as nn


class PDInverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X):
        
        X_inv = exact_inverse(X)
        ctx.save_for_backward(X_inv)

        return X_inv

    @staticmethod
    def backward(ctx, grad_output):
        (X_inv,) = ctx.saved_tensors
        grad_input = -X_inv @ grad_output @ X_inv
        grad_input = 0.5 * (grad_input + grad_input.transpose(-1, -2))
        return grad_input


class PDInverseModule(nn.Module):
    def forward(self, X):
        return PDInverse.apply(X)


def exact_inverse(F):

    try:
        L = torch.linalg.cholesky(F)
        Q = torch.cholesky_inverse(L)
    except torch.linalg.LinAlgError:
        Q = torch.linalg.inv(add_ridge(F,1e-10))

    Q = 0.5 * (Q + Q.t())
    return Q


def solve_with_cholesky(S, F):
    L = torch.linalg.cholesky(F)
    ST = S.T
    XT = torch.cholesky_solve(ST, L, upper=False)
    return XT.T

def add_ridge(F,reg):
    identity= torch.eye(F.shape[0], device=F.device, dtype=F.dtype)
    return F + reg * identity


