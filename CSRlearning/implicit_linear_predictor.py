import torch
import torch.nn as nn
from typing import Callable, Dict, Any, Optional, Sequence, Tuple
from linear_predictors import exact_inverse 

# ---------------------------
# Utility: ridge regression loss
# ---------------------------
def linear_probe_loss(X: torch.Tensor, Y: torch.Tensor, P: torch.Tensor, reg: float) -> torch.Tensor:
    """
    P: (k, k)
    X: (B, k)
    Y: (B, k)
    returns scalar loss = (1/(2B))||X P^T - Y||^2 + 0.5*reg*||P||^2
    """

    data_term = 0.5* torch.mean(torch.sum( (X @ P.T - Y)**2 , dim=1)) 
    data_term = data_term + 0.5* torch.mean(torch.sum( (Y @ P.T - X)**2 , dim=1)) 
    reg_term = 0.5 * reg * P.pow(2).sum()
    return data_term + reg_term


def adjoint_pred_loss(X: torch.Tensor, Y: torch.Tensor, P: torch.Tensor,  reg: float) -> torch.Tensor:
    """
    P: (k, k)
    X: (B, k)
    Y: (B, k)
    returns scalar loss = (1/(2B))||X P^T - Y||^2 + 0.5*reg*||P||^2
    """ 
    data_term = 0.5* torch.mean(torch.sum( (X @ P.T)**2 , dim=1)) 
    data_term = data_term + 0.5* torch.mean(torch.sum( (Y @ P.T)**2 , dim=1)) 
    data_term = data_term - torch.trace(P)
    reg_term = 0.5 * reg * P.pow(2).sum()
    loss = data_term + reg_term
    return loss

def auxiliary_loss( X: torch.Tensor, 
                    Y: torch.Tensor, 
                    P: torch.Tensor, Q: torch.Tensor,
                    reg: float) -> torch.Tensor:
    """
    P: (k, k)
    X: (B, k)
    Y: (B, k)
    returns scalar loss = (1/(2B))||X P^T - Y||^2 + 0.5*reg*||P||^2
    """ 
    B = X.shape[0]
    loss = 0.5*reg*(torch.mean(torch.sum(X**2, dim=1)) + torch.mean(torch.sum(Y**2, dim=1)))
    #err_pred_y = torch.einsum('kl,bl->bk',P,X)-Y
    #err_pred_x = torch.einsum('kl,bl->bk',P,Y)-X

    #loss = loss + 0.5* torch.einsum('kl,bl,bl->' ,Q,X,err_pred_y)/B
    #loss = loss + 0.5* torch.einsum('kl,bl,bl->' ,Q,Y,err_pred_x)/B
    loss = loss + 0.5* torch.mean(torch.sum((X@P.T-Y)*(X@Q.T), dim=1 ))
    loss = loss + 0.5* torch.mean(torch.sum((Y@P.T-X)*(Y@Q.T), dim=1 ))

    return loss
# ---------------------------
# Default inner solver: arbitrary optimizer (user-provided)
# ---------------------------

class IterativeLinearSolver(nn.Module):
    """
    Maintains P and optimizer state across calls.
    Intended for warm-started approximate inner solves.
    """
    def __init__(
        self,
        P_init: torch.Tensor,
        optimizer_ctor,
        reg: float,
        loss_fn=linear_probe_loss,
        val=1
    ):
        super().__init__()
        self.P = torch.nn.parameter.Parameter(data=P_init.clone().detach(), requires_grad=True)
        #self.P = P_init.clone().detach().to(device).requires_grad_(True)
        self.reg = reg
        self.loss_fn = loss_fn
        self.optimizer = optimizer_ctor([self.P])
        self.val = val
    def forward(self, X_det: torch.Tensor, Y_det: torch.Tensor):
        """
        Perform `steps` optimizer steps using detached features.
        """
        with torch.enable_grad():
            self.optimizer.zero_grad()
            loss = self.loss_fn(X_det, Y_det, self.P, self.reg)
            loss.backward()
            self.optimizer.step()
            #if self.val ==2:
            #    print(self.P.grad)
        return self.P.detach()


class IterativeDiffRegressor(nn.Module):
    def __init__(
        self,
        pred_init: torch.Tensor,
        adjoint_init: torch.Tensor,
        optimizer_ctor,
        reg: float,
    ):
        super().__init__()
        self.pred = torch.nn.parameter.Parameter(data=pred_init.clone().detach(), requires_grad=True)
        self.adjoint = torch.nn.parameter.Parameter(data=adjoint_init.clone().detach(), requires_grad=True)
        #self.P = P_init.clone().detach().to(device).requires_grad_(True)
        self.reg = reg
        self.pred_loss = linear_probe_loss
        self.adjoint_loss = adjoint_pred_loss
        self.optimizer = optimizer_ctor([self.pred, self.adjoint])
    def forward(self, X_det: torch.Tensor, Y_det: torch.Tensor):
        """
        Perform `steps` optimizer steps using detached features.
        """
        with torch.enable_grad():
            self.optimizer.zero_grad()
            loss = self.pred_loss(X_det, Y_det, self.pred, self.reg)
            loss = loss + self.adjoint_loss(X_det, Y_det, self.adjoint, self.reg)
            loss.backward()
            self.optimizer.step()
            #if self.val ==2:
            #    print(self.P.grad)
        return self.pred.detach(), self.adjoint.detach()




class MomentumRegressor(nn.Module):
    def __init__(
        self,
        pred_init: torch.Tensor,
        adjoint_init: torch.Tensor,
        reg: float,
        eta: float,
    ):
        super().__init__()
        self.inter_cov = torch.nn.parameter.Parameter(data=pred_init.clone().detach(), requires_grad=True)
        self.intra_cov = torch.nn.parameter.Parameter(data=adjoint_init.clone().detach(), requires_grad=True)
        #self.P = P_init.clone().detach().to(device).requires_grad_(True)
        self.reg = reg
        self.eta = eta
    def forward(self, X_det: torch.Tensor, Y_det: torch.Tensor):
        """
        Perform `steps` optimizer steps using detached features.
        """
        B = X_det.shape[0]
        inter_cov = (X_det.t()@Y_det)/B
        inter_cov =  inter_cov +  inter_cov.t()
        intra_cov = (X_det.t()@X_det)/B + (Y_det.t()@Y_det)/B


        self.intra_cov.data = (1-self.eta)*self.intra_cov.data + self.eta*intra_cov
        self.inter_cov.data = (1-self.eta)*self.inter_cov.data + self.eta*inter_cov

        self.adjoint = exact_inverse(self.intra_cov, self.reg)
        self.pred = inter_cov @ self.adjoint


        return self.pred.detach(), self.adjoint.detach()











