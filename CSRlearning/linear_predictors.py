import torch
import math
import torch.nn as nn


import torch

class PDInverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X):
        # X must be symmetric positive definite
        try:
            L = torch.linalg.cholesky(X)
            X_inv = torch.cholesky_inverse(L)
            ctx.save_for_backward(X_inv)
        except torch.linalg.LinAlgError:
            jitter = 1e-10 * torch.eye(X.shape[0], device=X.device, dtype=X.dtype)
            X_inv = torch.linalg.inv(X + jitter)
        X_inv = 0.5 * (X_inv + X_inv.t())

        return X_inv

    @staticmethod
    def backward(ctx, grad_output):
        (X_inv,) = ctx.saved_tensors

        # Gradient formula:
        # dL/dX = - X^{-1} @ dL/dY @ X^{-1}
        grad_input = -X_inv @ grad_output @ X_inv

        # Enforce symmetry (important for numerical stability)
        grad_input = 0.5 * (grad_input + grad_input.transpose(-1, -2))

        return grad_input







class PDInverseModule(nn.Module):
    def forward(self, X):
        return PDInverse.apply(X)


def exact_inverse(F, reg_c):
    F = F + reg_c*torch.eye(F.shape[0], device=F.device, dtype=F.dtype)

    try:
        # Sigma = L @ L.T
        L = torch.linalg.cholesky( F)
        
        # Q = inv(L.T) @ inv(L)
        Q = torch.cholesky_inverse(L)
        
    except torch.linalg.LinAlgError:
        # Backup: If Sigma is extremely ill-conditioned, add a tiny jitter
        # and use a general inverse.
        jitter = 1e-10 * torch.eye(F.shape[0], device=F.device, dtype=F.dtype)
        Q = torch.linalg.inv(F + jitter)
        
    # 3. Symmetry Reinforcement
    # Even with Cholesky, floating point noise can accumulate.
    Q = 0.5 * (Q + Q.t())
    return Q





import torch

def solve_with_cholesky(S, F, reg_c):
    """
    Computes S @ inv(F + c*I) using Cholesky decomposition.
    
    Args:
        S (Tensor): The matrix to be multiplied (n x d or m x d).
        F (Tensor): The PSD matrix (d x d).
        c (float): Regularization constant for numerical stability.
        
    Returns:
        Tensor: The result of S @ (F + cI)^{-1}.
    """
    # 1. Add regularization to the diagonal (F + cI)
    d = F.size(-1)
    identity = torch.eye(d, device=F.device, dtype=F.dtype)
    F_reg = F + (reg_c * identity)
    
    # 2. Compute Cholesky decomposition L (F_reg = L @ L.T)
    # upper=False returns the lower triangular matrix L
    L = torch.linalg.cholesky(F_reg)
    
    # 3. Solve (L @ L.T) X^T = S.T
    # S.T is used because cholesky_solve expects (A)X = B
    # We want X = S @ inv(A), so X.T = inv(A) @ S.T
    ST = S.T
    XT = torch.cholesky_solve(ST, L, upper=False)
    
    return XT.T

# Example usage:
# F = torch.randn(5, 5)
# F = F @ F.T  # Make it PSD
# S = torch.randn(2, 5)
# result = solve_with_cholesky(S, F, c=1e-5)








def newton_update(Q, F,reg_c, eta=0.01):
    F = F + reg_c*torch.eye(F.shape[0], device=F.device, dtype=F.dtype)
    #print(lol)
    return Q+ eta*(Q - Q@F@Q)


def newton_cholesky_update(D, L, F, reg_c, eta=0.1):
    """
    Performs one Newton-like update on the parameterization A = exp(D) + L.
    
    Args:
        D (torch.Tensor): Vector of size (d,) representing the diagonal log-factors.
        L (torch.Tensor): Strictly lower triangular matrix of size (d, d).
        F (torch.Tensor): The target matrix (e.g., E[xx^T] + cI) of size (d, d).
        eta (float): Learning rate (step size).
        
    Returns:
        Q (torch.Tensor): The current precision matrix estimate AA^T.
        D_new (torch.Tensor): Updated D.
        L_new (torch.Tensor): Updated L.
    """
    F = F + reg_c*torch.eye(F.shape[0], device=F.device, dtype=F.dtype)
    max_D = 0.5 * math.log(1.0 / reg_c)
    d = D.shape[0]
    identity = torch.eye(d, device=D.device, dtype=D.dtype)
    
    # 1. Construct A = diag(exp(D)) + L
    # We use exp(D) to guarantee positivity of the diagonal
    exp_D = torch.exp(D)
    A = torch.diag(exp_D) + L
    
    # 2. Compute the Factor Residual: M = I - A^T @ F @ A
    # This matrix measures how far Q^-1 is from F in the "A-space"
    # Note: For O(d^2), if F = xx^T + cI, expand this product!
    M = identity - A.T @ (F @ A)
    
    # 3. Project M into the space of lower triangular velocities H
    # H + H^T = M ensures the Newton step is satisfied.
    # Diagonal of H is 0.5 * diag(M).
    diag_M = torch.diag(M)
    H_diag = 0.5 * diag_M
    H_lower = torch.tril(M, diagonal=-1)
    H = torch.diag(H_diag) + H_lower
    
    # 4. Compute parameter updates
    # dD/dt = diag(H)
    # dL/dt = lower_strict(A @ H)
    delta_D = H_diag
    
    # A @ H is lower triangular because both are lower triangular
    AH = A @ H
    delta_L = torch.tril(AH, diagonal=-1)
    
    # 5. Apply update

    D_new = torch.clamp(D + eta * H_diag, max=max_D)

    L_new = L + eta * delta_L
    exp_D = torch.exp(D_new)
    A = torch.diag(exp_D) + L_new
    # 6. Compute current Q for the user
    Q = A @ A.T
    
    return Q, D_new, L_new




import torch

def preconditioned_first_order_step(D, L, F, c, eta=1e-3):
    """
    Preconditions theta updates to match Gradient Descent of Q.
    A = exp(D) + L
    Target: dot(Q) = I - FQ
    """
    d = D.shape[0]
    exp_D = torch.exp(D)
    A = torch.diag(exp_D) + L
    
    # 1. Compute Q_inv efficiently: Q_inv = (A A^T)^-1 = A^-T A^-1
    # We solve A @ Inv_A = I
    I = torch.eye(d, device=D.device)
    A_inv = torch.linalg.solve_triangular(A, I, upper=False)
    Q_inv = A_inv.T @ A_inv
    
    # 2. Compute the Preconditioned Residual W
    # W = Q_inv - A^T F A
    Phi = A.T @ (F @ A)
    W = Q_inv - Phi
    
    # 3. Project into H (Lower Triangular velocity)
    H_diag = 0.5 * torch.diag(W)
    H_lower = torch.tril(W, diagonal=-1)
    
    # 4. Apply to parameters
    # dot(D) = diag(H)
    D_new = D + eta * H_diag
    
    # dot(L) = lower_strict(A @ H)
    H = torch.diag(H_diag) + H_lower
    AH = A @ H
    L_new = L + eta * torch.tril(AH, diagonal=-1)
    
    # 5. Stability Truncation
    max_D = 0.5 * math.log(1.0 / c)
    D_new = torch.clamp(D_new, max=max_D)
    

    exp_D = torch.exp(D_new)
    A = torch.diag(exp_D) + L_new
    # 6. Compute current Q for the user
    Q = A @ A.T



    return Q, D_new, L_new


def plain_gradient_descent_step(D, L, F, c, eta=1e-4):
    """
    Plain Gradient Descent on parameters D and L.
    Objective: J = 0.5*tr(QFQ) - tr(Q)
    Parameterization: A = diag(exp(D)) + L, Q = AA^T
    """
    # 1. Reconstruct A and Q
    exp_D = torch.exp(D)
    A = torch.diag(exp_D) + L
    Q = A @ A.T
    
    # 2. Compute the Gradient w.r.t A
    # grad_A = 2 * (F @ Q - I) @ A
    # This is the standard Euclidean gradient in factor space.
    I = torch.eye(D.shape[0], device=D.device)
    grad_A = 2.0 * (F @ Q - I) @ A
    
    # 3. Chain Rule for D: grad_D = diag(grad_A) * exp(D)
    # This accounts for the exponential reparameterization.
    grad_D = torch.diag(grad_A) * exp_D
    
    # 4. Chain Rule for L: grad_L = strictly_lower(grad_A)
    grad_L = torch.tril(grad_A, diagonal=-1)
    
    # 5. Gradient Step
    D_new = D - eta * grad_D
    L_new = L - eta * grad_L
    
    # 6. Physical Constraint (Truncation)
    # Ensures Q doesn't explode beyond the regularization limit
    max_D = 0.5 * math.log(1.0 / c)
    D_new = torch.clamp(D_new, max=max_D)
    

    exp_D = torch.exp(D_new)
    A = torch.diag(exp_D) + L_new
    # 6. Compute current Q for the user
    Q = A @ A.T

    return Q, D_new, L_new


# import torch

def newton_cholesky_update(D, L, F, reg_c, eta=0.1, clip_thresh=0.5):
    """
    Newton update for A = exp(D) + L where F_t is a general matrix 
    (mini-batch or momentum estimate).
    """
    F = F + reg_c*torch.eye(F.shape[0], device=F.device, dtype=F.dtype)
    max_D = 0.5 * math.log(1./reg_c)
    d = D.shape[0]
    identity = torch.eye(d, device=D.device, dtype=D.dtype)


    # 0. Construct A = diag(exp(D)) + L
    # We use exp(D) to guarantee positivity of the diagonal
    exp_D = torch.exp(D)
    A = torch.diag(exp_D) + L
    
    # 1. Compute the Factor-Space Image: Phi = A^T @ F_t @ A
    # This matrix contains all the information about the local curvature.
    # Note: If F_t is dxd, this is O(d^3). 
    Phi = A.T @ (F @ A)
    
    # 2. Trace-Based Step Normalization (The Stability Gate)
    # This dynamically scales eta so the update never overshoots the 
    # quadratic bowl, even if F_t is poorly scaled from momentum.
    kappa = torch.trace(Phi)
    eta = eta / (1.0 + kappa)
    #print(f"effective eta: {eta}")
    
    # 3. Residual Matrix M = I - Phi
    M = identity - Phi
    
    # 4. Extract H components
    H_diag = 0.5 * torch.diag(M)
    H_lower = torch.tril(M, diagonal=-1)
    
    # 5. Riemannian Norm Clipping
    # Prevents "momentum shocks" from moving the parameters too far 
    # in one step.
    update_norm = torch.sqrt(torch.sum(H_diag**2) + torch.sum(H_lower**2))
    if update_norm > clip_thresh:

        scale = clip_thresh / update_norm
        H_diag *= scale
        H_lower *= scale
        #print(f"gradient clipping. Norm: {update_norm} greater than thresh: {clip_thresh}")    

    # 6. Apply Updates with your truncation
    
    D_new = torch.clamp(D + eta * H_diag, max=max_D)
    
    H = torch.diag(H_diag) + H_lower
    # A @ H is lower triangular; we update L using the strictly lower part.
    L_new = L + eta * torch.tril(A @ H, diagonal=-1)
    
    exp_D = torch.exp(D_new)
    A = torch.diag(exp_D) + L_new
    # 6. Compute current Q for the user
    Q = A @ A.T

    return Q, D_new, L_new


import torch

def newton_cholesky_update(D, L, F, c, eta=0.1, epsilon=0.05):
    """
    Implements Q -> Q + eta * Q(I - FQ) safely.
    Ensures rho(I - QF) < 1 is maintained.
    """
    F = F + c*torch.eye(F.shape[0], device=F.device, dtype=F.dtype)
    max_D = 0.5 * math.log(1./c)
    d = D.shape[0]
    A = torch.diag(torch.exp(D)) + L
    
    # 1. Calculate the Stability Metric
    # Phi = A^T F A, eig(Phi) = eig(QF)
    Phi = A.T @ (F @ A)
    
    # Trace is a conservative upper bound for lambda_max
    # We need lambda_max(QF) < 2/eta. 
    # We use a safety factor (1 - epsilon)
    R_u, S_u, _ = torch.linalg.svd(Phi)
    kappa = S_u[0]
    #kappa = torch.trace(Phi)
    
    limit = 1+ 1. / eta
    
    # 2. Robust Spectral Projection
    # If the current scale of A/Q is too large for the new F, 
    # we rescale A before the update.
    if kappa >= limit:

        kappa_0 = 1+ .5 / eta
        # Scale A down so that Tr(QF) = 1/eta (safely in the middle of the basin)
        scale = torch.sqrt(torch.tensor(kappa_0 / kappa))
        D = D + torch.log(scale)
        L = L * scale
        
        # Recompute A and Phi
        A = torch.diag(torch.exp(D)) + L
        Phi = A.T @ (F @ A)

    # 3. Perform the Damped Newton Update
    # Residual M = I - Phi
    M = torch.eye(d, device=D.device) - Phi
    
    # Velocity H = 0.5 * M (The 0.5 comes from the Q = AA^T derivative)
    H_diag = 0.5 * torch.diag(M)
    H_lower = torch.tril(M, diagonal=-1)
    
    # 4. Update Parameters
    # eta is applied here. Note: dD/dt = H_diag
    D_new = D + eta * H_diag
    
    # Clamp D based on the regularization floor c
    D_new = torch.clamp(D_new, max=max_D)
    
    # AH is lower triangular
    H = torch.diag(H_diag) + H_lower
    AH = A @ H
    L_new = L + eta * torch.tril(AH, diagonal=-1)
    
    exp_D = torch.exp(D_new)
    A = torch.diag(exp_D) + L_new
    # 6. Compute current Q for the user
    Q = A @ A.T


    return Q, D_new, L_new


import torch

def warm_start_first_oder(D, L, F, c, eta=0.01):
    """
    Polishes the warm-start (D, L) to move towards the new F.
    Uses trace-normalization to ensure the update is always stable.
    """
    # 1. Current state
    F = F + c*torch.eye(F.shape[0], device=F.device, dtype=F.dtype)
    max_D = 0.5 * math.log(1./c)


    A = torch.diag(torch.exp(D)) + L
    
    # 2. Compute Phi (Factor-space image of F)
    # This captures how well our warm-start fits the new F
    Phi = A.T @ (F @ A)
    
    # 3. Normalized Step Size (The "Secret Sauce")
    # By dividing by the trace, we ensure the effective 'lambda_max * eta' 
    # is always bounded, making divergence almost impossible.
    trace_phi = torch.trace(Phi)
    eta = eta / (1.0 + trace_phi / D.shape[0])
    
    # 4. First-Order Update (Safe & Stable)
    # M = I - Phi
    M = torch.eye(D.shape[0], device=D.device) - Phi
    
    # Map to D and L
    grad_D = torch.diag(M)
    grad_L = torch.tril(A @ M, diagonal=-1)
    
    # 5. Apply and Truncate
    D_new = D + eta * grad_D
    
    max_D = 0.5 * math.log(1.0/c)
    D_new = torch.clamp(D_new, max=max_D)
    L_new = L + eta * grad_L
    
    exp_D = torch.exp(D_new)
    A = torch.diag(exp_D) + L_new
    # 6. Compute current Q for the user
    Q = A @ A.T

    return Q, D_new, L_new




# Example Usage:
# d = 4
# D = torch.zeros(d)
# L = torch.zeros(d, d)
# F = torch.eye(d) * 2.0  # Target Q = 0.5 * I
# Q, D, L = newton_cholesky_update(D, L, F, eta=0.1)





def update_cov_old(data_gen, encoder_u, encoder_v, 
                S_0,F_0, eta_0, 
                n_opt_pred, 
                cold_start=False,
                symmetric_cov_uv=True,
                EMA=False):


    # --- UPDATE Q, S (LinPred) ---
    if cold_start:
        S = torch.zeros_like(S_0)
        F = torch.zeros_like(F_0)
    else:
        S = S_0
        F= F_0
    
    if EMA:
        def eta_step(step):
            return eta_0
    else:
        def eta_step(step):
            return 1./(step+1)

    with torch.no_grad():
        for iter_pred in range(n_opt_pred):
            batch_pred_x, batch_pred_y = next(data_gen)
            eta = eta_step(iter_pred)
            if symmetric_cov_uv:
                S = _update_cov_old(S, eta, 
                                batch_pred_y, encoder_v)
            else:
                S = _update_cov_old(S, eta, 
                                batch_pred_x, encoder_u,
                                batch_pred_y, encoder_v)
            F = _update_cov_old(F, eta, 
                                batch_pred_x, encoder_u)
    return S,F


def _update_cov_old(S, eta, x_b, encoder_u, y_b = None, encoder_v=None):

    with torch.no_grad():

        u = encoder_u(x_b) # [batch, k]
        batch_size = u.shape[0]
        if y_b is None:
            v = u
        else:
            if encoder_v is None:
                encoder_v = encoder_u
            v = encoder_v(y_b)   # [batch, k]

        # 1. Update S: S = (1 - eta_S)S + eta_S * Mean(v v^T)
        S = (1 - eta) * S + eta * torch.matmul(v.t(), u) / batch_size
        return S


# def lin_reg_full_solve(data_gen, n_opt_pred, encoder_u):
#     def eta_step(step):
#         return 1./(step+1)
#     def identity(x):
#         return x

#     with torch.no_grad():
#         for iter_pred in range(n_opt_pred):
#             batch_pred_x, batch_pred_y = next(data_gen)
#             eta = eta_step(iter_pred)
#             if iter_pred==0:
#                 u = encoder_u(batch_pred_x)
#                 k, d = u.shape[1], batch_pred_y.shape[1]
#                 S = torch.zeros(k,d, device=u.device)
#                 F = torch.zeros(d,d, device=u.device)


#             S = _update_cov_old(S, eta, 
#                             batch_pred_y, identity,
#                             batch_pred_x, encoder_u)
#             F = _update_cov_old(F, eta, 
#                             batch_pred_y, identity)
    
#         return  S@ exact_inverse(F,0.)








