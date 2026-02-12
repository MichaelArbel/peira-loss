import math
import torch


def newton_update(Q, F, reg_c, eta=0.01):
    F = F + reg_c * torch.eye(F.shape[0], device=F.device, dtype=F.dtype)
    return Q + eta * (Q - Q @ F @ Q)


def newton_cholesky_update(D, L, F, reg_c, eta=0.1):
    F = F + reg_c * torch.eye(F.shape[0], device=F.device, dtype=F.dtype)
    max_D = 0.5 * math.log(1.0 / reg_c)
    d = D.shape[0]
    identity = torch.eye(d, device=D.device, dtype=D.dtype)

    exp_D = torch.exp(D)
    A = torch.diag(exp_D) + L

    M = identity - A.T @ (F @ A)

    diag_M = torch.diag(M)
    H_diag = 0.5 * diag_M
    H_lower = torch.tril(M, diagonal=-1)
    H = torch.diag(H_diag) + H_lower

    D_new = torch.clamp(D + eta * H_diag, max=max_D)

    AH = A @ H
    L_new = L + eta * torch.tril(AH, diagonal=-1)
    exp_D = torch.exp(D_new)
    A = torch.diag(exp_D) + L_new
    Q = A @ A.T

    return Q, D_new, L_new


def preconditioned_first_order_step(D, L, F, c, eta=1e-3):
    d = D.shape[0]
    exp_D = torch.exp(D)
    A = torch.diag(exp_D) + L

    I = torch.eye(d, device=D.device)
    A_inv = torch.linalg.solve_triangular(A, I, upper=False)
    Q_inv = A_inv.T @ A_inv

    Phi = A.T @ (F @ A)
    W = Q_inv - Phi

    H_diag = 0.5 * torch.diag(W)
    H_lower = torch.tril(W, diagonal=-1)

    D_new = D + eta * H_diag

    H = torch.diag(H_diag) + H_lower
    AH = A @ H
    L_new = L + eta * torch.tril(AH, diagonal=-1)

    max_D = 0.5 * math.log(1.0 / c)
    D_new = torch.clamp(D_new, max=max_D)

    exp_D = torch.exp(D_new)
    A = torch.diag(exp_D) + L_new
    Q = A @ A.T

    return Q, D_new, L_new


def plain_gradient_descent_step(D, L, F, c, eta=1e-4):
    exp_D = torch.exp(D)
    A = torch.diag(exp_D) + L
    Q = A @ A.T

    I = torch.eye(D.shape[0], device=D.device)
    grad_A = 2.0 * (F @ Q - I) @ A

    grad_D = torch.diag(grad_A) * exp_D

    grad_L = torch.tril(grad_A, diagonal=-1)

    D_new = D - eta * grad_D
    L_new = L - eta * grad_L

    max_D = 0.5 * math.log(1.0 / c)
    D_new = torch.clamp(D_new, max=max_D)

    exp_D = torch.exp(D_new)
    A = torch.diag(exp_D) + L_new
    Q = A @ A.T

    return Q, D_new, L_new


def newton_cholesky_update_safe(D, L, F, c, eta=0.1, epsilon=0.05):
    F = F + c * torch.eye(F.shape[0], device=F.device, dtype=F.dtype)
    max_D = 0.5 * math.log(1.0 / c)
    d = D.shape[0]
    A = torch.diag(torch.exp(D)) + L

    Phi = A.T @ (F @ A)

    _, S_u, _ = torch.linalg.svd(Phi)
    kappa = S_u[0]

    limit = 1 + 1.0 / eta

    if kappa >= limit:
        kappa_0 = 1 + 0.5 / eta
        scale = torch.sqrt(torch.tensor(kappa_0 / kappa))
        D = D + torch.log(scale)
        L = L * scale

        A = torch.diag(torch.exp(D)) + L
        Phi = A.T @ (F @ A)

    M = torch.eye(d, device=D.device) - Phi

    H_diag = 0.5 * torch.diag(M)
    H_lower = torch.tril(M, diagonal=-1)

    D_new = D + eta * H_diag

    D_new = torch.clamp(D_new, max=max_D)

    H = torch.diag(H_diag) + H_lower
    AH = A @ H
    L_new = L + eta * torch.tril(AH, diagonal=-1)

    exp_D = torch.exp(D_new)
    A = torch.diag(exp_D) + L_new
    Q = A @ A.T

    return Q, D_new, L_new


def warm_start_first_order(D, L, F, c, eta=0.01):
    F = F + c * torch.eye(F.shape[0], device=F.device, dtype=F.dtype)
    max_D = 0.5 * math.log(1.0 / c)

    A = torch.diag(torch.exp(D)) + L

    Phi = A.T @ (F @ A)

    trace_phi = torch.trace(Phi)
    eta = eta / (1.0 + trace_phi / D.shape[0])

    M = torch.eye(D.shape[0], device=D.device) - Phi

    grad_D = torch.diag(M)
    grad_L = torch.tril(A @ M, diagonal=-1)

    D_new = D + eta * grad_D

    max_D = 0.5 * math.log(1.0 / c)
    D_new = torch.clamp(D_new, max=max_D)
    L_new = L + eta * grad_L

    exp_D = torch.exp(D_new)
    A = torch.diag(exp_D) + L_new
    Q = A @ A.T

    return Q, D_new, L_new


def update_covariance_old(
    data_gen,
    encoder_u,
    encoder_v,
    S_0,
    F_0,
    eta_0,
    n_opt_pred,
    cold_start=False,
    symmetric_cov_uv=True,
    EMA=False,
):
    if cold_start:
        S = torch.zeros_like(S_0)
        F = torch.zeros_like(F_0)
    else:
        S = S_0
        F = F_0

    if EMA:
        def eta_step(step):
            return eta_0
    else:
        def eta_step(step):
            return 1.0 / (step + 1)

    with torch.no_grad():
        for iter_pred in range(n_opt_pred):
            batch_pred_x, batch_pred_y = next(data_gen)
            eta = eta_step(iter_pred)
            if symmetric_cov_uv:
                S = _update_covariance_old(S, eta, batch_pred_y, encoder_v)
            else:
                S = _update_covariance_old(
                    S, eta, batch_pred_x, encoder_u, batch_pred_y, encoder_v
                )
            F = _update_covariance_old(F, eta, batch_pred_x, encoder_u)
    return S, F


def _update_covariance_old(S, eta, x_b, encoder_u, y_b=None, encoder_v=None):
    with torch.no_grad():
        u = encoder_u(x_b)
        batch_size = u.shape[0]
        if y_b is None:
            v = u
        else:
            if encoder_v is None:
                encoder_v = encoder_u
            v = encoder_v(y_b)

        S = (1 - eta) * S + eta * torch.matmul(v.t(), u) / batch_size
        return S
