import torch
import torch.nn as nn
from typing import Callable, Dict, Any, Optional, Sequence, Tuple
from .linalg_utils import exact_inverse, add_ridge, solve_with_cholesky
from .objectives import linear_probe_loss, adjoint_pred_loss
from factory import instantiate
# ---------------------------
# Default inner solver: arbitrary optimizer (user-provided)
# ---------------------------

class IterativeDiffRegressor(nn.Module):
    def __init__(
        self,
        pred_init: torch.Tensor,
        adjoint_init: torch.Tensor,
        optimizer_ctor,
        scheduler_ctor, 
        reg: float,
        symmetric=True,
        with_adjoint=False
    ):
        super().__init__()
        self.pred = torch.nn.parameter.Parameter(data=pred_init.clone().detach(), requires_grad=True)
        self.adjoint = torch.nn.parameter.Parameter(data=adjoint_init.clone().detach(), requires_grad=True)
        #self.P = P_init.clone().detach().to(device).requires_grad_(True)
        self.reg = reg
        self.pred_loss = linear_probe_loss
        self.adjoint_loss = adjoint_pred_loss
        self.optimizer = optimizer_ctor([self.pred, self.adjoint])
        self.scheduler = scheduler_ctor(self.optimizer)
        self.symmetric = symmetric
        self.with_inverse = with_adjoint

    def forward(self, X_det: torch.Tensor, Y_det: torch.Tensor):
        """
        Perform `steps` optimizer steps using detached features.
        """
        with torch.enable_grad():
            self.optimizer.zero_grad()
            loss = self.pred_loss(X_det, Y_det, self.pred, self.reg, 
                                 symmetric=self.symmetric)
            if self.with_adjoint:
                loss = loss + self.adjoint_loss(X_det, Y_det, self.adjoint, self.reg)
            loss.backward()
            self.optimizer.step()
            #if self.val ==2:
            #    print(self.P.grad)
        if self.with_adjoint:
            return self.pred.detach(), self.adjoint.detach()
        else: 
            return self.pred.detach()



class MomentumRegressor(nn.Module):
    def __init__(
        self,
        pred_init: torch.Tensor,
        adjoint_init: torch.Tensor,
        optimizer_ctor,
        scheduler_ctor,
        reg: float,
        symmetric: bool=True,
        with_adjoint: bool=True
    ):
        super().__init__()
        self.inter_cov = torch.nn.parameter.Parameter(data=pred_init.clone().detach(), requires_grad=True)
        self.intra_cov = torch.nn.parameter.Parameter(data=adjoint_init.clone().detach(), requires_grad=True)
        #self.P = P_init.clone().detach().to(device).requires_grad_(True)
        self.reg = reg
        self.optimizer = optimizer_ctor([self.intra_cov, self.inter_cov])
        self.scheduler = scheduler_ctor(self.optimizer)
        self.symmetric = symmetric
        self.with_inverse = with_adjoint
    def forward(self, X_det: torch.Tensor, Y_det: torch.Tensor):
        """
        Perform `steps` optimizer steps using detached features.
        """
        B = X_det.shape[0]
        inter_cov = (Y_det.t()@X_det)/B
        intra_cov = (X_det.t()@X_det)/B

        if self.symmetric:
            inter_cov =  inter_cov +  inter_cov.t()
            intra_cov = intra_cov + (Y_det.t()@Y_det)/B
        self.optimizer.zero_grad()
        self.intra_cov.grad = self.intra_cov.data-intra_cov
        self.inter_cov.grad = self.inter_cov.data-inter_cov
        self.optimizer.step()
        
        if self.with_adjoint:
            self.adjoint = exact_inverse(add_ridge(self.intra_cov, self.reg))
            self.pred = inter_cov @ self.adjoint
            return self.pred.detach(), self.adjoint.detach()
        else:
            self.pred = solve_with_cholesky(inter_cov, add_ridge(self.intra_cov, self.reg))
            return self.pred.detach()



def get_regressor_spec(regressor, lr, T_max, symmetric, with_adjoint):
    regressor_spec = {
        "name": regressor,
        "kwargs": {
            "symmetric": symmetric,
            "with_adjoint": with_adjoint,
        },
        "optimizer": {"name": "torch.optim.SGD", "kwargs": {"lr": lr}},
        "scheduler": {
            "name": "torch.optim.lr_scheduler.CosineAnnealingLR",
            "kwargs": {"T_max": T_max},
        },
    }
    return regressor_spec
    



def build_regressor(spec, latent_dim, reg):
    regressor = spec.get("name", None)
    if regressor:
        regressor_cls = instantiate(regressor)
    else:
        return None
    
    k = latent_dim
    pred_init = torch.zeros(k, k)
    adjoint_init = torch.zeros(k, k)
    kwargs = spec.get("kwargs", {})
    optimizer_spec = spec.get("optimizer", {})
    scheduler_spec = spec.get("scheduler", {})

    if "name" not in optimizer_spec:
        raise ValueError("regressor spec is missing optimizer.name")

    optimizer_cls = instantiate(optimizer_spec.get("name"))
    optimizer_ctor = lambda param: optimizer_cls(param, **optimizer_spec.get("kwargs", {}))

    if scheduler_spec:
        scheduler_cls = instantiate(scheduler_spec.get("name"))
        scheduler_ctor = lambda optimizer: scheduler_cls(optimizer, **scheduler_spec.get("kwargs", {}))
    else:
        scheduler_ctor = lambda optimizer: None

    return regressor_cls(
        pred_init,
        adjoint_init,
        optimizer_ctor=optimizer_ctor,
        scheduler_ctor=scheduler_ctor,
        reg=reg,
        symmetric=kwargs.get("symmetric", True),
        with_adjoint=kwargs.get("with_adjoint", False),
    )

