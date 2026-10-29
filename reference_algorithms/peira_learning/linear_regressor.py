import importlib
import torch
import torch.nn as nn
from typing import Callable, Dict, Any, Optional, Sequence, Tuple
from .linalg_utils import exact_inverse, add_ridge, solve_with_cholesky
from .objectives import linear_probe_loss, adjoint_pred_loss
# ---------------------------
# Default inner solver: arbitrary optimizer (user-provided)
# ---------------------------

class IterativeDiffRegressor(nn.Module):
    def __init__(
        self,
        pred_init: torch.Tensor,
        adjoint_init: torch.Tensor,
        reg: float,
        optimizer_ctor = lambda param: torch.optim.SGD(param, lr=0.01),
        scheduler_ctor = lambda optimizer: None, 
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
        self.with_adjoint = with_adjoint

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
        reg: float,
        optimizer_ctor = lambda param: torch.optim.SGD(param, lr=0.01),
        scheduler_ctor = lambda optimizer: None, 
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
        if isinstance(self.optimizer,torch.optim.AdamW):
            self.reg = 0.
        self.symmetric = symmetric
        self.with_adjoint = with_adjoint
        self.is_first= True
        self.last_stats = {}
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
        if self.is_first:
            self.intra_cov.data = intra_cov
            self.inter_cov.data = inter_cov
            self.is_first = False
        else:
            self.optimizer.zero_grad()
            self.intra_cov.grad = self.intra_cov.data-intra_cov
            self.inter_cov.grad = self.inter_cov.data-inter_cov
            self.optimizer.step()
            self.scheduler.step()

        if self.with_adjoint:
            self.adjoint = exact_inverse(add_ridge(self.intra_cov, self.reg))
            self.pred = self.inter_cov @ self.adjoint
            pred  = inter_cov @ exact_inverse(add_ridge(intra_cov, self.reg))
            inter_gap = self.inter_cov.detach() - inter_cov.detach()
            intra_gap = self.intra_cov.detach() - intra_cov.detach()
            pred_gap = self.pred.detach() - pred.detach()
            self.last_stats = {
                "peira_trace_running": float(torch.trace(self.pred.detach()).item()),
                "peira_trace_batch": float(torch.trace(pred.detach()).item()),
                "peira_trace_gap": float(
                    abs(torch.trace(self.pred.detach()).item() - torch.trace(pred.detach()).item())
                ),
                "peira_pred_gap_rel_fro": float(
                    torch.norm(pred_gap).item() / (torch.norm(pred.detach()).item() + 1e-12)
                ),
                "peira_inter_cov_gap_rel_fro": float(
                    torch.norm(inter_gap).item() / (torch.norm(inter_cov.detach()).item() + 1e-12)
                ),
                "peira_intra_cov_gap_rel_fro": float(
                    torch.norm(intra_gap).item() / (torch.norm(intra_cov.detach()).item() + 1e-12)
                ),
                "peira_inner_lr": float(self.optimizer.param_groups[0]["lr"]),
                "peira_reg": float(self.reg),
            }

            return self.pred.detach(), self.adjoint.detach()
        else:
            #try:
            #    self.pred = solve_with_cholesky(self.inter_cov, add_ridge(self.intra_cov, self.reg))
            
            #except:
            adjoint = exact_inverse(add_ridge(self.intra_cov, self.reg))
            self.pred = self.inter_cov @ adjoint
            #print(lol)
            return self.pred.detach()
