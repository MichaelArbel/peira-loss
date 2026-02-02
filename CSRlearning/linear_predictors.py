import torch



def exact_inverse(F, reg_c):
    try:
        # Sigma = L @ L.T
        L = torch.linalg.cholesky(F + reg_c*torch.eye(F.shape[0], device=F.device, dtype=F.dtype) )
        
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





def newton_update(Q, F,reg_c, eta=0.01):
    F = F + reg_c*torch.eye(F.shape[0], device=F.device, dtype=F.dtype)
    Q += eta*(Q@Q - Q@F@Q)
    return Q
















