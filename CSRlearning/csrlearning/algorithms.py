import torch
import torch.nn as nn

from .objectives import (
    cse_auxiliary_loss,
    cse_loss,
    eckhart_young_loss,
    l2_penalty,
    linear_probe_loss,
)
from .linear_regressor import get_regressor_spec, build_regressor
from .core_types import AlgorithmState, DataSource

class BaseAlgorithm(nn.Module):
    def __init__(self):
        super().__init__()
        self.stop_grad = False

    def configure(self, encoders):
        return [], []

    def forward(self, X, Y, state):
        raise NotImplementedError

    def update_internal_state(self, data_gen: DataSource, encoders) -> AlgorithmState:
        return AlgorithmState(
            extras={"consumed_batches": 0},
            stop_grad=self.stop_grad,
        )




class UnconstrainedCCAAlgorithm(BaseAlgorithm):

    def forward(self, X, Y, state):
        return eckhart_young_loss(X, Y)

class CSEAlgorithm(BaseAlgorithm):
    def __init__(self, reg):
        super().__init__()
        self.reg = reg

class SSLAlgorithm(BaseAlgorithm):
    def __init__(self, reg):
        super().__init__()
        self.stop_grad = True
        self.reg = reg

    def configure(self, encoders):
        self._weights = [param for param in encoders[0].parameters()]
        return [], []

class CSEExactRegressionAlgorithm(CSEAlgorithm):
    def forward(self, X, Y, state):
        return cse_loss(X, Y, None, self.reg)





class StochasticCompositionalGradientAlgorithm(BaseAlgorithm):
    """
    Models that contain a linear regressor that is optimized 
    during call to update_internal_state method
    """
    def __init__(self, latent_dim, reg, n_iter, regressor_spec):
        super().__init__()
        self.reg = reg
        self.latent_dim = latent_dim
        self.n_iter = n_iter
        self.regressor = build_regressor(regressor_spec, 
                                        self.latent_dim,
                                        self.reg)
        self.encoders = None

    def update_internal_state(self, data_gen: DataSource, encoders) -> AlgorithmState:
        state = super().update_internal_state(data_gen, encoders)
        for _ in range(self.n_iter):
            with torch.no_grad():
                data = next(data_gen)
                outputs = tuple([encoders[i](data[i]) for i in range(len(data))])
            self.pred = self.regressor(outputs[0].detach(), outputs[1].detach())
        consumed = self.n_iter
        state.extras["consumed_batches"] = consumed

        if isinstance(self.pred, tuple) and len(self.pred) == 2:
            state.pred = self.pred[0]
            state.adjoint = self.pred[1]
        else:
            state.pred = self.pred
        return state




class CSEStoCompGrad(StochasticCompositionalGradientAlgorithm, CSEAlgorithm):
    def __init__(self, latent_dim, reg, n_iter, T_max, lr):

        regressor = "MomentumRegressor"
        regressor_spec = get_regressor_spec(regressor, 
                                            lr, 
                                            T_max, 
                                            symmetric=True, 
                                            with_adjoint=True)
        super().__init__(latent_dim=latent_dim, 
                         reg=reg, 
                         n_iter= n_iter, 
                         regressor_spec= regressor_spec)

    def forward(self, X, Y, state):
        pred, adjoint_pred = state.pred, state.adjoint
        return cse_auxiliary_loss(X, Y, pred, adjoint_pred, self.reg)




class CSEBilevelStoGrad(StochasticCompositionalGradientAlgorithm, CSEAlgorithm):
    def __init__(self, latent_dim, reg, n_iter, T_max, lr):

        regressor = "IterativeDiffRegressor"
        regressor_spec = get_regressor_spec(regressor, 
                                            lr, 
                                            T_max, 
                                            symmetric=True, 
                                            with_adjoint=True)
        super().__init__(latent_dim=latent_dim, 
                         reg=reg, 
                         n_iter= n_iter, 
                         regressor_spec= regressor_spec)

    def forward(self, X, Y, state):
        pred, adjoint_pred = state.pred, state.adjoint
        return cse_auxiliary_loss(X, Y, pred, adjoint_pred, self.reg)



class CSEBilevelStoGradJoint(StochasticCompositionalGradientAlgorithm,CSEAlgorithm):
    def __init__(self, latent_dim, reg, T_max, lr):
        regressor = "IterativeDiffRegressor"
        regressor_spec = get_regressor_spec(regressor, 
                                            lr, 
                                            T_max, 
                                            symmetric=True, 
                                            with_adjoint=True)
        super().__init__(latent_dim=latent_dim, 
                         reg=reg, 
                         n_iter= 1, 
                         regressor_spec= regressor_spec)


    def configure(self, encoders):
        return [self.regressor.optimizer], [self.regressor.scheduler]

    def update_internal_state(self, data_gen: DataSource, encoders) -> AlgorithmState:
        return AlgorithmState(extras={"consumed_batches": 0}, stop_grad=self.stop_grad)

    def forward(self, X, Y, state):
        pred = self.regressor.pred
        adjoint = self.regressor.adjoint
        X_det, Y_det = X.detach(), Y.detach()

        pred_loss = self.regressor.pred_loss(X_det, Y_det, pred, self.reg)
        adjoint_loss = self.regressor.adjoint_loss(X_det, Y_det, adjoint, self.reg)
        loss_aux = cse_auxiliary_loss(X, Y, pred.detach(), adjoint.detach(), self.reg)
        return pred_loss + adjoint_loss + loss_aux


class SSLStoCompSemiGrad(StochasticCompositionalGradientAlgorithm,SSLAlgorithm):
    def __init__(self, latent_dim, reg, n_iter, T_max, lr):

        regressor = "csrlearning.linear_regressor.MomentumRegressor"
        regressor_spec = get_regressor_spec(regressor, 
                                            lr, 
                                            T_max, 
                                            symmetric=False, 
                                            with_adjoint=False)
        super().__init__(latent_dim=latent_dim, 
                         reg=reg, 
                         n_iter= n_iter, 
                         regressor_spec= regressor_spec)

    def forward(self, X, Y, state):
        loss = linear_probe_loss(X, Y, state.pred, self.reg, symmetric=False)
        loss = loss + 0.5 * self.reg * l2_penalty(self._weights)
        return loss


class SSLBilevelStoSemiGrad(StochasticCompositionalGradientAlgorithm,SSLAlgorithm):
    def __init__(self, latent_dim, reg, n_iter, T_max, lr):

        regressor = "csrlearning.linear_regressor.IterativeDiffRegressor"
        regressor_spec = get_regressor_spec(regressor, 
                                            lr, 
                                            T_max, 
                                            symmetric=False, 
                                            with_adjoint=False)
        super().__init__(latent_dim=latent_dim, 
                 reg=reg, 
                 n_iter= n_iter, 
                 regressor_spec= regressor_spec)

    def forward(self, X, Y, state):
        loss = linear_probe_loss(X, Y, state.pred, self.reg, symmetric=False)
        loss = loss + 0.5 * self.reg * l2_penalty(self._weights)
        return loss



class SSLStoSemiGradJoint(StochasticCompositionalGradientAlgorithm,SSLAlgorithm):
    def __init__(self, latent_dim, reg, T_max, lr):
        regressor = "IterativeDiffRegressor"
        regressor_spec = get_regressor_spec(regressor, 
                                            lr, 
                                            T_max, 
                                            symmetric=False, 
                                            with_adjoint=False)
        super().__init__(latent_dim=latent_dim, 
                 reg=reg, 
                 n_iter= 1, 
                 regressor_spec= regressor_spec)


    def configure(self, encoders):
        self._weights = [param for param in encoders[0].parameters()]
        return [self.regressor.optimizer], [self.regressor.scheduler]

    def update_internal_state(self, data_gen: DataSource, encoders) -> AlgorithmState:
        return AlgorithmState(extras={"consumed_batches": 0}, stop_grad=self.stop_grad)

    def forward(self, X, Y, state):
        pred = self.regressor.pred
        loss = linear_probe_loss(X, Y, pred, self.reg, symmetric=False)
        loss = loss + 0.5 * self.reg * l2_penalty(self._weights)
        return loss
