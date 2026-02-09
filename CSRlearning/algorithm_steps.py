import torch
import torch.nn as nn

from algorithms import (
    compute_cse_objective,
    compute_regression_loss,
    compute_running_average_field,
    compute_semi_gradient_ssl_loss,
)
from evaluation import (
    compute_cse_spectrum,
    compute_semigrad_spectrum,
    compute_unconstrained_cca_spectrum,
)
from implicit_linear_predictor import IterativeDiffRegressor, MomentumRegressor, auxiliary_loss
from linear_predictors import PDInverseModule, solve_with_cholesky


class AlgorithmStep(nn.Module):
    def __init__(self):
        super().__init__()

    def configure_model(self, model):
        pass

    def compute_spectrum(self, model, data_gen):
        raise NotImplementedError

    def step(self, model, data_gen):
        raise NotImplementedError


class CanonicalSubspaceEmbeddingExactAlgorithm(AlgorithmStep):
    def __init__(self):
        super().__init__()
        self.inverse = PDInverseModule()

    def compute_spectrum(self, model, data_gen):
        return compute_cse_spectrum(model, data_gen)

    def step(self, model, data_gen):
        m = model
        batch_cse_x, batch_cse_y = next(data_gen)

        m.opt_u.zero_grad()
        m.opt_v.zero_grad()
        u_active = m.encoder_u(batch_cse_x)
        v_active = m.encoder_v(batch_cse_y)

        B = u_active.shape[0]
        inter_cov = torch.einsum("mk,ml->kl", v_active, u_active) / B
        cov_u = torch.einsum("mk,ml->kl", u_active, u_active) / B
        cov_v = torch.einsum("mk,ml->kl", v_active, v_active) / B

        cov = cov_v + cov_u + m.reg_c * torch.eye(cov_u.shape[0], device=cov_u.device, dtype=cov_u.dtype)
        inv_cov = self.inverse(cov)
        loss = -0.5 * torch.sum((inter_cov + inter_cov.t()) * inv_cov)
        loss = loss + 0.5 * m.reg_c * (torch.sum(v_active ** 2) + torch.sum(u_active ** 2)) / B
        loss.backward()
        #print(lol)


        m.running_grad = compute_running_average_field(m.running_grad, m.encoder_u.weight.grad, m.eta)
        grad_norm = torch.norm(m.encoder_u.weight.grad)
        running_grad_norm = torch.norm(m.running_grad)

        m.opt_u.step()
        m.opt_v.step()

        return {
            "loss_reg": loss.item(),
            "loss_aux": loss.item(),
            "loss_CSE": loss.item(),
            "grad_norm": grad_norm.item(),
            "running_grad_norm": running_grad_norm.item(),
            "Q_norm": torch.norm(m.Q).item(),
            "S_norm": torch.norm(m.S).item(),
        }


class CanonicalSubspaceEmbeddingBilevelAlgorithm(AlgorithmStep):
    def __init__(self):
        super().__init__()
        self.regressor = None
        self.rep_loss = auxiliary_loss
        self.inverse = PDInverseModule()

    def configure_model(self, model):
        optimizer_ctor = lambda params: torch.optim.SGD(params, lr=model.eta)
        self.regressor = IterativeDiffRegressor(
            torch.zeros(model.k, model.k),
            torch.zeros(model.k, model.k),
            optimizer_ctor=optimizer_ctor,
            reg=model.reg_c,
        )


        # self.regressor = MomentumRegressor(
        #     torch.zeros(model.k, model.k),
        #     torch.zeros(model.k, model.k),
        #     reg=model.reg_c,
        #     eta=model.eta
        # )

    def compute_spectrum(self, model, data_gen):
        return compute_cse_spectrum(model, data_gen)

    def step(self, model, data_gen):
        m = model
        for _ in range(m.n_opt_pred):
            with torch.no_grad():
                batch_cse_x, batch_cse_y = next(data_gen)
                X = m.encoder_u(batch_cse_x)
                Y = m.encoder_v(batch_cse_y)
            pred, adj_pred = self.regressor(X.detach(), Y.detach())

        m.opt_u.zero_grad()
        m.opt_v.zero_grad()
        batch_cse_x, batch_cse_y = next(data_gen)
        X = m.encoder_u(batch_cse_x)
        Y = m.encoder_v(batch_cse_y)

        loss_aux = self.rep_loss(X, Y, pred, adj_pred, m.reg_c)

        loss = 0.5 * torch.trace(pred) - 0.5 * m.reg_c * (
            torch.mean(torch.sum(X ** 2, dim=1)) + torch.mean(torch.sum(Y ** 2, dim=1))
        )

        loss_aux.backward()
        

        # grad_U =1.* m.encoder_u.weight.grad
        # grad_V =1.* m.encoder_v.weight.grad
        

        # m.opt_u.zero_grad()
        # m.opt_v.zero_grad()

        # u_active = m.encoder_u(batch_cse_x)
        # v_active = m.encoder_v(batch_cse_y)
        # B = u_active.shape[0]
        # inter_cov = torch.einsum("mk,ml->kl"  ,v_active,u_active)/B
        # cov_u = torch.einsum("mk,ml->kl", u_active,u_active)/B
        # cov_v =torch.einsum("mk,ml->kl", v_active,v_active)/B
        
        # cov = cov_v+ cov_u + m.reg_c*torch.eye(cov_u.shape[0], device=cov_u.device, dtype=cov_u.dtype)
        # inv_cov = self.inverse(cov)
        # loss = -0.5*torch.sum((inter_cov + inter_cov.t())*inv_cov)   
        # loss = loss + 0.5*m.reg_c*(torch.sum(v_active**2) + torch.sum(u_active**2))/B



        # loss.backward()

        # grad_UU =1.* m.encoder_u.weight.grad
        # grad_VV =1.* m.encoder_v.weight.grad        


        # print(f"grad_U err: {torch.norm(grad_UU-grad_U)}")
        # print(f"grad_U err: {torch.norm(grad_VV-grad_V)}")

        # print(lol)






        m.running_grad = compute_running_average_field(m.running_grad, m.encoder_u.weight.grad, m.eta)
        grad_norm = torch.norm(m.encoder_u.weight.grad)
        running_grad_norm = torch.norm(m.running_grad)

        m.opt_u.step()
        m.opt_v.step()

        return {
            "loss_reg": loss.item(),
            "loss_aux": loss_aux.item(),
            "loss_CSE": loss.item(),
            "grad_norm": grad_norm.item(),
            "running_grad_norm": running_grad_norm.item(),
            "Q_norm": torch.norm(m.Q).item(),
            "S_norm": torch.norm(m.S).item(),
        }


class CanonicalSubspaceEmbeddingBilevelJointAlgorithm(AlgorithmStep):
    def __init__(self):
        super().__init__()
        self.regressor = None
        self.rep_loss = auxiliary_loss

    def configure_model(self, model):
        optimizer_ctor = lambda params: torch.optim.SGD(params, lr=model.eta)
        self.regressor = IterativeDiffRegressor(
            torch.zeros(model.k, model.k),
            torch.zeros(model.k, model.k),
            optimizer_ctor=optimizer_ctor,
            reg=model.reg_c,
        )

    def compute_spectrum(self, model, data_gen):
        return compute_cse_spectrum(model, data_gen)

    def step(self, model, data_gen):
        m = model
        m.opt_u.zero_grad()
        m.opt_v.zero_grad()
        self.regressor.optimizer.zero_grad()

        batch_cse_x, batch_cse_y = next(data_gen)
        X = m.encoder_u(batch_cse_x)
        Y = m.encoder_v(batch_cse_y)
        pred = self.regressor.pred
        adjoint = self.regressor.adjoint
        X_det, Y_det = X.detach(), Y.detach()

        pred_loss = self.regressor.pred_loss(X_det, Y_det, pred, m.reg_c)
        adjoint_loss = self.regressor.adjoint_loss(X_det, Y_det, adjoint, m.reg_c)
        loss_aux = self.rep_loss(X, Y, pred.detach(), adjoint.detach(), m.reg_c)
        total_loss = pred_loss + adjoint_loss + loss_aux
        loss = 0.5 * torch.trace(pred) - 0.5 * m.reg_c * (
            torch.mean(torch.sum(X ** 2, dim=1)) + torch.mean(torch.sum(Y ** 2, dim=1))
        )

        total_loss.backward()

        m.running_grad = compute_running_average_field(m.running_grad, m.encoder_u.weight.grad, m.eta)
        grad_norm = torch.norm(m.encoder_u.weight.grad)
        running_grad_norm = torch.norm(m.running_grad)

        m.opt_u.step()
        m.opt_v.step()
        self.regressor.optimizer.step()

        return {
            "loss_reg": loss.item(),
            "loss_aux": loss_aux.item(),
            "loss_CSE": loss.item(),
            "grad_norm": grad_norm.item(),
            "running_grad_norm": running_grad_norm.item(),
            "Q_norm": torch.norm(m.Q).item(),
            "S_norm": torch.norm(m.S).item(),
        }


class SemiGradientSSLAlgorithm(AlgorithmStep):
    def __init__(self):
        super().__init__()
        self.predictor = None
        self.opt_pred = None
        self.scheduler_pred = None

    def configure_model(self, model):
        model.encoder_v = model.encoder_u
        model.sym_cov = False
        self.predictor = nn.Linear(model.k, model.k, bias=False)
        self.opt_pred = torch.optim.SGD(self.predictor.parameters(), lr=model.eta)
        self.scheduler_pred = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_pred, t_max=model.t_max, eta_min=0.000001
        )

    def compute_spectrum(self, model, data_gen):
        return compute_semigrad_spectrum(model, data_gen)

    def step(self, model, data_gen):
        m = model

        if m.full_solve_v:
            with torch.no_grad():
                m.update_covariance_estimates(data_gen, sym_cov=False)
                predictor = solve_with_cholesky(m.S, m.F, m.reg_c)
        else:
            for _ in range(m.n_opt_pred):
                self.opt_pred.zero_grad()
                with torch.no_grad():
                    batch_cse_x, batch_cse_y = next(data_gen)
                    v_fixed = m.encoder_v(batch_cse_y)
                    u_fixed = m.encoder_u(batch_cse_x)
                loss_aux = compute_semi_gradient_ssl_loss(u_fixed, v_fixed, self.predictor.weight, m.reg_c, m.encoder_u)
                loss_aux = loss_aux + 0.5 * m.reg_c * torch.sum(self.predictor.weight ** 2)
                loss_aux.backward()
                self.opt_pred.step()
            predictor = self.predictor.weight.data

        batch_cse_x, batch_cse_y = next(data_gen)
        m.opt_u.zero_grad()
        u_active = m.encoder_u(batch_cse_x)
        with torch.no_grad():
            v_fixed = m.encoder_v(batch_cse_y)
        loss_aux = compute_semi_gradient_ssl_loss(u_active, v_fixed, predictor, m.reg_c, m.encoder_u)
        loss_aux.backward()
        m.running_grad = compute_running_average_field(m.running_grad, m.encoder_u.weight.grad, m.eta)
        grad_norm = torch.norm(m.encoder_u.weight.grad)
        running_grad_norm = torch.norm(m.running_grad)

        m.opt_u.step()
        if not m.full_solve_v:
            self.scheduler_pred.step()

        m.step_schedulers()

        return {
            "loss_reg": 0.0,
            "loss_aux": loss_aux.item(),
            "loss_CSE": 0.0,
            "grad_norm": grad_norm.item(),
            "running_grad_norm": running_grad_norm.item(),
            "Q_norm": torch.norm(m.Q).item(),
            "S_norm": torch.norm(m.S).item(),
        }


class UnconstrainedCCAAlgorithm(AlgorithmStep):
    def compute_spectrum(self, model, data_gen):
        return compute_unconstrained_cca_spectrum(model, data_gen)

    def step(self, model, data_gen):
        m = model
        batch_cse_x, batch_cse_y = next(data_gen)

        m.opt_u.zero_grad()
        m.opt_v.zero_grad()
        u_active = m.encoder_u(batch_cse_x)
        v_active = m.encoder_v(batch_cse_y)

        B = u_active.shape[0]
        inter_cov = torch.einsum("mk,mk->", u_active, v_active) / B
        cov_u = torch.einsum("mk,ml->kl", u_active, u_active) / B
        cov_v = torch.einsum("mk,ml->kl", v_active, v_active) / B
        intra_cov = torch.sum((cov_u + cov_v) ** 2)

        loss = -4.0 * inter_cov + intra_cov
        loss.backward()
        m.running_grad = compute_running_average_field(m.running_grad, m.encoder_u.weight.grad, m.eta)
        grad_norm = torch.norm(m.encoder_u.weight.grad)
        running_grad_norm = torch.norm(m.running_grad)

        m.opt_u.step()
        m.opt_v.step()
        m.step_schedulers()

        return {
            "loss_reg": 0.0,
            "loss_aux": loss.item(),
            "loss_CSE": loss.item(),
            "grad_norm": grad_norm.item(),
            "running_grad_norm": running_grad_norm.item(),
            "Q_norm": torch.norm(m.Q).item(),
            "S_norm": torch.norm(m.S).item(),
        }
