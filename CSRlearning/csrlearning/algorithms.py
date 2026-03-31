import torch
import torch.nn as nn

from .objectives import (
    cse_auxiliary_loss,
    cse_loss,
    eckhart_young_loss,
    l2_penalty,
    linear_probe_loss,
    exact_linear_probe_loss,
    eckhart_young_sto_loss
)

from typing import Any, Dict

from dataclasses import dataclass, field

from .linalg_utils import solve_with_cholesky, add_ridge



@dataclass(frozen=False)
class AlgorithmState:
    pred: Any = None
    adjoint: Any = None
    extras: Dict[str, Any] = field(default_factory=dict)
    stop_grad: bool = False



class RegScheduler:
    def __init__(self,algorithm, scheduler_ctor):
        self.algorithm = algorithm
                    # dummy parameter, never actually used by your model
        dummy = torch.nn.Parameter(torch.tensor(0.0))

        # store alpha in the optimizer LR field
        self.opt = torch.optim.SGD([dummy], lr=algorithm.reg)
        self.scheduler = scheduler_ctor(self.opt)

    def step(self, epoch):

        self.opt.step()
        self.scheduler.step(epoch)
        self.algorithm.reg = self.opt.param_groups[0]["lr"]
        self.algorithm.regressor.reg = self.opt.param_groups[0]["lr"]



class BaseAlgorithm(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.stop_grad = False
        self.tied_encoders = False

    def configure(self, encoders):
        return [(None,None)]

    def inner_steps(self) -> int:
        return 0

    def forward(self, X, Y, state):
        raise NotImplementedError

    def update_internal_state(self, X,Y) -> AlgorithmState:
        return self.get_internal_state()
    def get_internal_state(self):
        return AlgorithmState(
            extras={"consumed_batches": 0},
            stop_grad=self.stop_grad,
        )



class UnconstrainedCCAAlgorithm(BaseAlgorithm):

    def forward(self, X, Y, state):
        return eckhart_young_loss(X, Y)


class EYStoAlgorithm(UnconstrainedCCAAlgorithm):

    def forward(self, X, Y, state):
        return eckhart_young_sto_loss(X, Y)


class CSEAlgorithm(BaseAlgorithm):
    def __init__(self, reg, **kwargs):
        super().__init__(reg=reg, **kwargs)
        self.reg = reg

    def forward(self, X, Y, state):
        pred, adjoint_pred = state.pred, state.adjoint
        return cse_auxiliary_loss(X, Y, pred, adjoint_pred, self.reg)




class SSLAlgorithmParam(BaseAlgorithm):
    def __init__(self, reg, **kwargs):
        super().__init__(reg=reg, **kwargs)
        self.stop_grad = True
        self.tied_encoders = True
        self.reg = reg

    def configure(self, encoders):
        self._weights = [param for param in encoders[0].parameters()]
        return super().configure(encoders)

    def forward(self, X, Y, state):
        loss = linear_probe_loss(X, Y, state.pred, self.reg, symmetric=False)
        loss = loss + 0.5 * self.reg * l2_penalty(self._weights)
        return loss


class SSLAlgorithm(BaseAlgorithm):
    def __init__(self, reg, **kwargs):
        super().__init__(reg=reg, **kwargs)
        self.stop_grad = True
        self.tied_encoders = True
        self.reg = reg

    def configure(self, encoders):
        self._weights = [param for param in encoders[0].parameters()]
        return super().configure(encoders)

    def forward(self, X, Y, state):
        loss = linear_probe_loss(X, Y, state.pred, self.reg, symmetric=False)
        loss = loss + 0.5 * self.reg * torch.mean(torch.sum(X**2,dim=1))
        return loss



class CSEExactRegressionAlgorithm(CSEAlgorithm):
    def forward(self, X, Y, state):
        return cse_loss(X, Y, None, self.reg)




class SSLExactSemiGrad(SSLAlgorithm):
    def forward(self, X, Y, state):
        loss = exact_linear_probe_loss(X, Y, self.reg, symmetric=False)
        loss = loss + 0.5 * self.reg * l2_penalty(self._weights)
        return loss



class StochasticCompositionalGradientAlgorithm(BaseAlgorithm):
    """
    Models that contain a linear regressor that is optimized 
    during call to update_internal_state method
    """
    def __init__(
        self,
        regressor_cl,
        reg,
        n_iter,
        latent_dim,
        optimizer_ctor,
        scheduler_ctor,
        symmetric,
        with_adjoint, 
        reg_scheduler_ctor = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_iter = n_iter
        self.reg = reg
        if reg_scheduler_ctor:
            self.reg_scheduler = RegScheduler(self, reg_scheduler_ctor)
        else:
            self.reg_scheduler = None

        self.regressor = regressor_cl(
            pred_init=torch.zeros(latent_dim,latent_dim),
            adjoint_init=torch.zeros(latent_dim,latent_dim),
            reg=self.reg,
            optimizer_ctor=optimizer_ctor,
            scheduler_ctor=scheduler_ctor,
            symmetric=symmetric,
            with_adjoint=with_adjoint
        )

    def configure(self, encoders):
        return [(None,self.regressor.scheduler)]


    def inner_steps(self) -> int:
        assert self.n_iter>0
        return self.n_iter

    def get_internal_state(self):
        state = super().get_internal_state()
        consumed = 1
        state.extras["consumed_batches"] = consumed

        if isinstance(self.pred, tuple) and len(self.pred) == 2:
            state.pred = self.pred[0]
            state.adjoint = self.pred[1]
        else:
            state.pred = self.pred
        return state

    def update_internal_state(self, X,Y) -> AlgorithmState:
        self.pred = self.regressor(X.detach(), Y.detach())
        return self.get_internal_state()
        


class BilevelStoGradJoint(StochasticCompositionalGradientAlgorithm):
    def __init__(self, 
                 latent_dim, 
                 reg, 
                 symmetric, 
                 with_adjoint, 
                 lr=0.01, 
                 T_max=20000,
                 lr_min=0.0,
                 **kwargs):
        n_iter = 1
        from .linear_regressor import IterativeDiffRegressor
        optimizer_ctor = lambda params: torch.optim.SGD(params, lr=lr)
        scheduler_ctor = lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(opt, eta_min= lr_min, 
                                                                                    T_max=T_max)
        super().__init__(IterativeDiffRegressor, 
                         reg, 
                         n_iter, 
                         latent_dim,
                         optimizer_ctor, 
                         scheduler_ctor,
                         symmetric=symmetric,
                         with_adjoint=with_adjoint,
                         **kwargs
                         )

    def inner_steps(self) -> int:
        return 0

    def configure(self, encoders):
        return [(self.regressor.optimizer, self.regressor.scheduler)]

    def get_internal_state(self, X,Y) -> AlgorithmState:
        return AlgorithmState(extras={"consumed_batches": 0}, stop_grad=self.stop_grad)






class StoCompGrad(StochasticCompositionalGradientAlgorithm):
    def __init__(self, 
                 n_iter, 
                 latent_dim, 
                 reg, symmetric, 
                 with_adjoint, 
                 lr=0.01, 
                 T_max=20000,
                 lr_min=0.0,
                 reg_min=None,
                 **kwargs):
        from .linear_regressor import MomentumRegressor
        optimizer_ctor = lambda params: torch.optim.SGD(params, lr=lr)
        scheduler_ctor = lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max,
                                                                                     eta_min=lr_min)
        
        if reg_min is not None:
            reg_scheduler_ctor = lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max,
                                                                             eta_min=reg_min)
        else:
            reg_scheduler_ctor = None

        super().__init__(MomentumRegressor, 
                         reg, 
                         n_iter, 
                         latent_dim,
                         optimizer_ctor, 
                         scheduler_ctor,
                         symmetric=symmetric,
                         with_adjoint=with_adjoint,
                         reg_scheduler_ctor=reg_scheduler_ctor,
                         **kwargs
                         )



class BilevelStoGrad(StochasticCompositionalGradientAlgorithm):
    def __init__(self, 
                n_iter, 
                latent_dim, 
                reg, 
                symmetric, 
                with_adjoint, 
                lr=0.01, 
                T_max=20000,
                **kwargs):

        from .linear_regressor import IterativeDiffRegressor
        optimizer_ctor = lambda params: torch.optim.SGD(params, lr=lr)
        scheduler_ctor = lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max)
        super().__init__(IterativeDiffRegressor, 
                         reg, 
                         n_iter, 
                         latent_dim,
                         optimizer_ctor, 
                         scheduler_ctor,
                         symmetric=symmetric,
                         with_adjoint=with_adjoint,
                         **kwargs
                         )




class CSEStoCompGrad_old(CSEAlgorithm, StoCompGrad):
    def __init__(self, 
                 n_iter, 
                 latent_dim, 
                 reg, 
                 lr=0.01, 
                 T_max=20000,
                 lr_min=0.0,
                 **kwargs):
        super().__init__(
            n_iter=n_iter,
            latent_dim=latent_dim,
            reg=reg,
            symmetric=True,
            with_adjoint=True,
            lr=lr,
            T_max=T_max,
            lr_min=lr_min,
            **kwargs
            )




class CSEBilevelStoGrad(CSEAlgorithm, BilevelStoGrad):
    def __init__(self, 
                n_iter, 
                latent_dim, 
                reg, 
                lr=0.01, 
                T_max=20000,
                lr_min=0.0,
                **kwargs):
        super().__init__( 
                        n_iter= n_iter, 
                        latent_dim=latent_dim, 
                        reg=reg, 
                        symmetric = True, 
                        with_adjoint = True, 
                        lr=lr, 
                        T_max=T_max,
                        lr_min=lr_min,
                        **kwargs)





class CSEStoCompGrad(CSEAlgorithm, StoCompGrad):
    def __init__(self, 
                 latent_dim, 
                 n_iter, 
                 reg, 
                 lr=0.01, 
                 T_max=20000,
                 lr_min=0.0,
                 reg_min=None,
                 **kwargs):
        super().__init__(
            n_iter=n_iter,
            latent_dim=latent_dim,
            reg=reg,
            symmetric=True,
            with_adjoint=True,
            lr=lr,
            T_max=T_max,
            lr_min=lr_min,
            reg_min=reg_min,
            **kwargs
            )

    def forward(self, X, Y, state=None):
        with torch.autocast(device_type=X.device.type, enabled=False):
            X_fp32 = X.detach().float()
            Y_fp32 = Y.detach().float()

            for _ in range(self.inner_steps()):
                state = self.update_internal_state(X_fp32, Y_fp32)

        pred, adjoint_pred = state.pred, state.adjoint
        peira_loss = cse_loss(X,Y, pred, self.reg)

        return {'loss': cse_auxiliary_loss(X, Y, pred, adjoint_pred, self.reg),
                'peira_loss': peira_loss}










class CSEBilevelStoGradJoint(CSEAlgorithm,BilevelStoGradJoint):
    def __init__(self, 
                 latent_dim, 
                 reg,
                 ema=1., 
                 lr=0.01, 
                 T_max=20000,
                 **kwargs):
        self.ema = ema
        if ema <1.:
            self.ema_pred = None
            self.ema_adjoint = None
        super().__init__(
                latent_dim=latent_dim,
                reg= reg, 
                symmetric= True, 
                with_adjoint= True, 
                lr=lr, 
                T_max=T_max,
                **kwargs)

    def forward(self, X, Y, state=None):
        pred = self.regressor.pred
        adjoint = self.regressor.adjoint
        if self.ema<1.:
            if self.ema_pred is None:
                self.ema_pred, self.ema_adjoint = pred.detach(), adjoint.detach()
            else:
                self.ema_pred = (1-self.ema)*self.ema_pred + self.ema*pred.detach()
                self.ema_adjoint = (1-self.ema)*self.ema_adjoint + self.ema*adjoint.detach()
            loss_aux = cse_auxiliary_loss(X, Y, self.ema_pred, self.ema_adjoint, self.reg)
        else:
            loss_aux = cse_auxiliary_loss(X, Y, pred.detach(), adjoint.detach(), self.reg)
        
        X_det, Y_det = X.detach(), Y.detach()
        pred_loss = self.regressor.pred_loss(X_det, Y_det, pred, self.reg)
        adjoint_loss = self.regressor.adjoint_loss(X_det, Y_det, adjoint, self.reg)

        total_loss = pred_loss + adjoint_loss + loss_aux
        peira_loss = cse_loss(X,Y, pred, self.reg)
        return {"loss": total_loss,
                "adjoint_loss": adjoint_loss,
                "pred_loss": pred_loss,
                "aux_loss": loss_aux,
                'peira_loss':peira_loss} 

class SSLStoCompSemiGrad(SSLAlgorithm, StoCompGrad):
                                                                                                    
    def __init__(self, 
                     latent_dim, 
                     n_iter, 
                     reg, 
                     lr=0.01, 
                     T_max=20000,
                     lr_min=0.0,
                     reg_min=None,
                     **kwargs):
        super().__init__(
            n_iter=n_iter,
            latent_dim=latent_dim,
            reg=reg,
            symmetric=False,
            with_adjoint=False,
            lr=lr,
            T_max=T_max,
            lr_min=lr_min,
            reg_min=reg_min,
            **kwargs
            )

    def forward(self, X, Y, state=None):
        with torch.autocast(device_type=X.device.type, enabled=False):
            X_fp32 = X.detach().float()
            Y_fp32 = Y.detach().float()

            for _ in range(self.inner_steps()):
                state = self.update_internal_state(X_fp32, Y_fp32)

        pred, adjoint_pred = state.pred, state.adjoint
        loss = linear_probe_loss(X, Y.detach(), pred, self.reg, symmetric=False)
        loss = loss + 0.5 * self.reg * torch.mean(torch.sum(X**2,dim=1))

        return {'loss': loss}






class SSLBilevelStoSemiGrad(SSLAlgorithm, BilevelStoGrad):
    def __init__(self, 
                n_iter, 
                latent_dim, 
                reg, 
                lr=0.01, 
                T_max=20000,
                **kwargs):
        super().__init__( 
                        n_iter= n_iter, 
                        latent_dim=latent_dim, 
                        reg=reg, 
                        symmetric = False, 
                        with_adjoint = False, 
                        lr=lr, 
                        T_max=T_max,
                        **kwargs)






class SSLStoSemiGradJoint(SSLAlgorithm,BilevelStoGradJoint):
    def __init__(self, 
                latent_dim, 
                reg, 
                lr=0.01, 
                T_max=20000,
                **kwargs):
        super().__init__(
                        latent_dim=latent_dim,
                        reg= reg, 
                        symmetric= False, 
                        with_adjoint= False, 
                        lr=lr, 
                        T_max=T_max,
                        **kwargs)

    def configure(self, encoders):
        self._weights = [param for param in encoders[0].parameters()]
        return [(self.regressor.optimizer, self.regressor.scheduler)]

    def forward(self, X, Y, state=None):
        pred = self.regressor.pred
        loss = linear_probe_loss(X, Y, pred, self.reg, symmetric=False)
        loss = loss + 0.5 * self.reg * torch.mean(torch.sum(X**2,dim=1))
        return {"loss": loss}

    


