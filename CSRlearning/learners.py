import torch
import torch.nn as nn
from functools import partial

from algorithms import cosine_momentum_schedule
from evaluation import compute_alignment_metrics
from utils import instantiate


def _update_covariance_ema(S, eta, v, u=None):
    batch_size = v.shape[0]
    if u is None:
        u = v
    # 1. Update S: S = (1 - eta_S)S + eta_S * Mean(v v^T)
    if S is None:
        S = torch.matmul(v.t(), u) / batch_size
    else:
        S = (1 - eta) * S + eta * torch.matmul(v.t(), u) / batch_size
    return S


class CSELearningModel(nn.Module):
    def __init__(
        self,
        encoder_u,
        encoder_v,
        algorithm_name,
        latent_dim=10,
        eta=0.1,
        reg_c=1.0,
        lr_u=0.001,
        lr_v=0.001,
        n_opt_v=10,
        n_opt_pred=1,
        n_accum_pred=10,
        update_mode_cold_start=False,
        full_solve_v=False,
        t_max=20000,
        params_q=None,
    ):
        super().__init__()
        if params_q is None:
            params_q = {
                "update_rule": "exact_inverse",
                "momentum": False,
                "cold_start": False,
                "kwargs": None,
            }

        self.encoder_u = encoder_u
        self.encoder_v = encoder_v

        self.k = latent_dim

        self.register_buffer("Q", torch.eye(latent_dim))
        self.register_buffer("S", torch.zeros(latent_dim, latent_dim))
        self.register_buffer("F", torch.zeros(latent_dim, latent_dim))
        if params_q["update_rule"] == "newton_cholesky_update":
            self.register_buffer("D", torch.zeros(latent_dim))
            self.register_buffer("L", torch.zeros(latent_dim, latent_dim))

        self.eta = eta
        self.reg_c = reg_c

        self.n_opt_v = n_opt_v
        self.n_opt_pred = n_opt_pred
        self.n_accum_pred = n_accum_pred
        self.t_max = t_max
        self.lr_u = lr_u
        self.lr_v = lr_v

        self.opt_u = torch.optim.SGD(self.encoder_u.parameters(), lr=lr_u)
        self.opt_v = torch.optim.SGD(self.encoder_v.parameters(), lr=lr_v)
        self.scheduler_u = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_u, t_max=t_max, eta_min=0.000001
        )
        self.scheduler_v = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_v, t_max=t_max, eta_min=0.000001
        )
        self.scheduler_eta = partial(
            cosine_momentum_schedule, t_max=t_max, m_start=self.eta, m_end=0.001
        )

        self.update_mode_cold_start = update_mode_cold_start
        self.running_grad = None
        self.first_update = True
        self.full_solve_v = full_solve_v
        self.sym_cov = None
        self.params_q = params_q
        self.counter = 0
        if params_q["kwargs"] is None:
            params_q["kwargs"] = {}

        algorithm_class = instantiate(algorithm_name)
        self.algorithm = algorithm_class()
        self.algorithm.configure_model(self)

    def compute_target_spectrum(self, data_gen):
        Sx = data_gen.info["S_1"]
        Sy = data_gen.info["S_2"]
        sigma = data_gen.info["sigma"]
        target_spectrum = torch.abs(Sx * Sy) / (
            torch.sqrt(Sx ** 2 + sigma ** 2) * torch.sqrt(Sy ** 2 + sigma ** 2)
        )
        target_spectrum, _ = torch.sort(target_spectrum, descending=True)
        return target_spectrum

    def update_covariance_estimates(self, data_gen, sym_cov):
        with torch.no_grad():
            S = torch.zeros_like(self.S)
            F = torch.zeros_like(self.F)
            for iter_accum in range(self.n_accum_pred):
                batch_pred_x, batch_pred_y = next(data_gen)
                u_code = self.encoder_u(batch_pred_x)
                v_code = self.encoder_v(batch_pred_y)
                eta = 1.0 / (iter_accum + 1)
                if sym_cov:
                    S = _update_covariance_ema(S, eta, v_code)
                else:
                    S = _update_covariance_ema(S, eta, v_code, u_code)
                F = _update_covariance_ema(F, eta, u_code)

            if not self.update_mode_cold_start or self.first_update:
                self.S = S
                self.F = F
                self.first_update = False
            else:
                self.S = (1 - self.eta) * self.S + self.eta * S
                self.F = (1 - self.eta) * self.F + self.eta * F

        return S, F

    def update_linear_predictor(self, data_gen):
        with torch.no_grad():
            if self.params_q["cold_start"]:
                self.Q = torch.eye(self.Q.shape[0], device=self.Q.device, dtype=self.Q.dtype)
            for _ in range(self.n_opt_pred):
                S, F = self.update_covariance_estimates(data_gen, sym_cov=self.sym_cov)
                if self.params_q["momentum"]:
                    F = self.F

                self.Q = self.compute_precision_matrix(F)

    def compute_precision_matrix(self, F):
        if self.params_q["update_rule"] == "exact_inverse":
            from linear_predictors import exact_inverse

            return exact_inverse(F, self.reg_c)
        if self.params_q["update_rule"] == "newton_update":
            from linear_predictors import newton_update

            return newton_update(self.Q, F, self.reg_c, **self.params_q["kwargs"])
        if self.params_q["update_rule"] == "newton_cholesky_update":
            from linear_predictors import newton_cholesky_update

            Q_new, self.D, self.L = newton_cholesky_update(
                self.D, self.L, F, self.reg_c, **self.params_q["kwargs"]
            )
            return Q_new
        raise NotImplementedError

    def forward_backward(self, data_gen):
        return self.algorithm.step(self, data_gen)

    def compute_spectrum(self, data_gen):
        return self.algorithm.compute_spectrum(self, data_gen)

    def compute_alignment(self, data_gen, num_batches=10):
        return compute_alignment_metrics(self, data_gen, num_batches=num_batches)

    def reset_parameters(self, mode="standard", scale=1e-5):
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    if mode == "near_zero":
                        nn.init.normal_(m.weight, mean=0.0, std=scale)
                    elif mode == "orthogonal":
                        nn.init.normal_(m.weight, mean=0.0, std=1.0)
                        m.weight.data, _ = torch.linalg.qr(m.weight.data, mode="reduced")

                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        self.Q.copy_(torch.eye(self.k, device=self.Q.device))
        self.S.copy_(torch.zeros(self.k, self.k, device=self.S.device))
        self.F.copy_(torch.zeros(self.k, self.k, device=self.F.device))
        self.first_update = True

    def step_schedulers(self):
        self.scheduler_u.step()
        self.scheduler_v.step()
        self.counter += 1
        self.eta = self.scheduler_eta(self.counter)
