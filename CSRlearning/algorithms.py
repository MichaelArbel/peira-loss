import torch
import torch.nn as nn

from objectives import (
    cse_auxiliary_loss,
    cse_loss,
    eckhart_young_loss,
    l2_penalty,
    linear_probe_loss,
)
from evaluation import (
    compute_cse_spectrum,
    compute_semigrad_spectrum,
    compute_unconstrained_cca_spectrum,
)
from linear_regressor import get_regressor_spec, build_regressor
from utils import encode
from csr_types import AlgorithmState

class BaseAlgorithm(nn.Module):
    def __init__(self):
        super().__init__()
        self.stop_grad = False

    def configure(self, encoders, optimizers, schedulers):
        pass

    def compute_spectrum(self):
        raise NotImplementedError

    def compute_loss(self, X, Y, state):
        raise NotImplementedError

    def update_internal_state(self, data_gen):
        return AlgorithmState(extras={"consumed_batches": 0})




class UnconstrainedCCAAlgorithm(BaseAlgorithm):

    def compute_spectrum(self, cov_spectrum, target_spectrum_info):
        target_spectrum = target_spectrum_info.spectrum
        return compute_unconstrained_cca_spectrum(cov_spectrum, target_spectrum)

    def compute_loss(self, X, Y, state):
        return eckhart_young_loss(X, Y)

class CSEAlgorithm(BaseAlgorithm):
    def __init__(self, reg):
        super().__init__()
        self.reg = reg
    def compute_spectrum(self, cov_spectrum, target_spectrum_info):
        target_spectrum = target_spectrum_info.spectrum
        return compute_cse_spectrum(cov_spectrum,target_spectrum , self.reg)

class SSLAlgorithm(BaseAlgorithm):
    def __init__(self, reg):
        super().__init__()
        self.stop_grad = True
        self.reg = reg
    def compute_spectrum(self, cov_spectrum, target_spectrum_info):
        return compute_semigrad_spectrum(cov_spectrum, target_spectrum_info, self.reg)



class CSEExactRegressionAlgorithm(CSEAlgorithm):
    def compute_loss(self, X, Y, state):
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

    def configure(self, encoders, optimizers, schedulers):
        if hasattr(self.regressor, "scheduler"):
            schedulers.append(self.regressor.scheduler)
        self.encoders = encoders

    def update_internal_state(self,data_gen):
        for _ in range(self.n_iter):
            with torch.no_grad():
                data = next(data_gen)
                outputs = encode(data, self.encoders)
            self.pred = self.regressor(outputs.x1.detach(), outputs.x2.detach())
        consumed = self.n_iter
        if isinstance(self.pred, tuple) and len(self.pred) == 2:
            return AlgorithmState(pred=self.pred[0], adjoint=self.pred[1], extras={"consumed_batches": consumed})
        return AlgorithmState(pred=self.pred, extras={"consumed_batches": consumed})




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

    def compute_loss(self, X, Y, state):
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

    def compute_loss(self, X, Y, state):
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


    def configure(self, encoders, optimizers, schedulers):
        super().configure(encoders, optimizers, schedulers)
        optimizers.append(self.regressor.optimizer)

    def update_internal_state(self, data_gen):
        return AlgorithmState(extras={"consumed_batches": 0})

    def compute_loss(self, X, Y, state):
        pred = self.regressor.pred
        adjoint = self.regressor.adjoint
        X_det, Y_det = X.detach(), Y.detach()

        pred_loss = self.regressor.pred_loss(X_det, Y_det, pred, self.reg)
        adjoint_loss = self.regressor.adjoint_loss(X_det, Y_det, adjoint, self.reg)
        loss_aux = cse_auxiliary_loss(X, Y, pred.detach(), adjoint.detach(), self.reg)
        return pred_loss + adjoint_loss + loss_aux


class SSLStoCompSemiGrad(StochasticCompositionalGradientAlgorithm,SSLAlgorithm):
    def __init__(self, latent_dim, reg, n_iter, T_max, lr):

        regressor = "MomentumRegressor"
        regressor_spec = get_regressor_spec(regressor, 
                                            lr, 
                                            T_max, 
                                            symmetric=False, 
                                            with_adjoint=False)
        super().__init__(latent_dim=latent_dim, 
                         reg=reg, 
                         n_iter= n_iter, 
                         regressor_spec= regressor_spec)

    def compute_loss(self, X, Y, state):
        loss = linear_probe_loss(X, Y, state.pred, self.reg, symmetric=False)
        loss = loss + 0.5 * self.reg * l2_penalty(self.encoders[0])
        return loss


class SSLBilevelStoSemiGrad(StochasticCompositionalGradientAlgorithm,SSLAlgorithm):
    def __init__(self, latent_dim, reg, n_iter, T_max, lr):

        regressor = "IterativeDiffRegressor"
        regressor_spec = get_regressor_spec(regressor, 
                                            lr, 
                                            T_max, 
                                            symmetric=False, 
                                            with_adjoint=False)
        super().__init__(latent_dim=latent_dim, 
                 reg=reg, 
                 n_iter= n_iter, 
                 regressor_spec= regressor_spec)

    def compute_loss(self, X, Y, state):
        loss = linear_probe_loss(X, Y, state.pred, self.reg, symmetric=False)
        loss = loss + 0.5 * self.reg * l2_penalty(self.encoders[0])
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


    def configure(self, encoders, optimizers, schedulers):
        super().configure(encoders, optimizers, schedulers)
        optimizers.append(self.regressor.optimizer)

    def update_internal_state(self, data_gen):
        return AlgorithmState(extras={"consumed_batches": 0})

    def compute_loss(self, X, Y, state):
        pred = self.regressor.pred
        loss = linear_probe_loss(X, Y, pred, self.reg, symmetric=False)
        loss = loss + 0.5 * self.reg * l2_penalty(self.encoders[0])
        return loss
