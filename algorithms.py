import torch
import torch.nn as nn

import torch
import torch.nn as nn




        
def compute_inverse(F, reg_c):
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


def compute_l_reg(u, v):
    # Placeholder for L_reg(U, V)
    return 0.5*torch.nn.functional.mse_loss(v, u)

def compute_l_aux(u, v, Q, S, reg_c):
    # Placeholder for L_aux(U, V, Q, S)
    # Implements a form of the Contrastive Spectral Objective
    # tr(U^T Q S Q U) or similar logic derived from your CSE research
    Qu = torch.einsum( "kl,ml->mk", Q,u)

    loss = (1./2)* torch.einsum("mk,mk->", (Qu @ S), Qu)/ u.shape[0]
    loss += 0.5*reg_c* torch.einsum("mk,mk->",u,u)/ u.shape[0]
    loss += - torch.einsum("mk,mk->", Qu, v)/u.shape[0]

    return loss

def compute_l_CSE(Q,S,u, reg_c):
    return (1./2)* torch.sum(S*Q) - 0.5* reg_c*torch.einsum("mk,mk->",u,u)/ u.shape[0]


def compute_SG_SSL(u,v, Q, S, reg_c,model):
    
    l2_penalty = 0
    for param in model.parameters():
        l2_penalty += torch.sum(param ** 2)
    pred = torch.einsum('rk,kl,bl->br',S,Q,u)
    return 0.5*torch.nn.functional.mse_loss(pred, v) + 0.5*reg_c*l2_penalty


def compute_spectrum(data_gen, 
                        encoder,
                        view,
                        num_batches=10):

    # 2. Estimate F = E[U(x)U(x)^T] using several fresh batches

    for iteration in range(num_batches):
        x_batch, y_batch = next(data_gen)
        if view=="x":
            data = x_batch
        elif view=="y":
            data = y_batch
        u = encoder(data) # [batch, k]
        if iteration==0:
            k = u.shape[1]
            Euu = torch.zeros((k,k), device=u.device)

        # Center u to compute the actual covariance if necessary
        Euu += torch.matmul(u.t(), u) / (u.shape[0] * num_batches)
    
    # 3. SVD of F to find its principal directions (eigenvectors)
    # Since F is a covariance matrix, it is symmetric; eigh is also an option.
    # U_f contains eigenvectors in columns.
    R_u, S_u, _ = torch.linalg.svd(Euu)

    out = {"R_u": R_u,  "S_u": S_u, "Euu":Euu}
    return out


def compute_alignment_func(D, U):
    # 4. Compute Alignment: cosine_sim(u_i, D u_i)
    # We transform each eigenvector of F by D
    # U_f: [k, k] (columns are eigenvectors u_i)
    # D_ui: [k, k] (columns are the transformed vectors D @ u_i)
    D_ui = torch.matmul(D, U) 

    # Calculate cosine similarity for each pair of columns
    # cos(theta) = (u_i . D_ui) / (||u_i|| * ||D_ui||)
    # Since u_i are eigenvectors of SVD, ||u_i|| = 1.
    dot_products = torch.sum(U * D_ui, dim=0)
    norms_D_ui = torch.norm(D_ui, dim=0)
    
    return dot_products / (norms_D_ui + 1e-8)



def compute_avg_field(avg_field_old, new_field, eta):
    if avg_field_old is None:
        return new_field
    else:
        return (1-eta)*avg_field_old + eta*new_field





def update_cov(data_gen, encoder_u, encoder_v, 
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
                S = _update_cov(S, eta, 
                                batch_pred_y, encoder_v)
            else:
                S = _update_cov(S, eta, 
                                batch_pred_x, encoder_u,
                                batch_pred_y, encoder_v)
            F = _update_cov(F, eta, 
                                batch_pred_x, encoder_u)
    return S,F


def _update_cov(S, eta, x_b, encoder_u, y_b = None, encoder_v=None):

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


def lin_reg_full_solve(data_gen, n_opt_pred, encoder_u):
    def eta_step(step):
        return 1./(step+1)
    def identity(x):
        return x

    with torch.no_grad():
        for iter_pred in range(n_opt_pred):
            batch_pred_x, batch_pred_y = next(data_gen)
            eta = eta_step(iter_pred)
            if iter_pred==0:
                u = encoder_u(batch_pred_x)
                k, d = u.shape[1], batch_pred_y.shape[1]
                S = torch.zeros(k,d, device=u.device)
                F = torch.zeros(d,d, device=u.device)


            S = _update_cov(S, eta, 
                            batch_pred_y, identity,
                            batch_pred_x, encoder_u)
            F = _update_cov(F, eta, 
                            batch_pred_y, identity)
    
        return  S@ compute_inverse(F,0.)





class RepLearner(nn.Module):
    def __init__(self, encoder_u, encoder_v, 
                        latent_dim=10, 
                        eta= 0.1,
                        reg_c = 1.0,
                        lr_theta= 0.001,
                        lr_psi= 0.001,
                        n_opt_v= 10,
                        n_opt_pred=10,
                        update_mode_cold_start=False,
                        full_solve_v=False):

        super().__init__()
        # Representations
        self.encoder_u = encoder_u
        self.encoder_v = encoder_v
        
        # Latent dim (k) for covariance matrices
        self.k = latent_dim
        
        # Auxiliary matrices Q and S (Initialized as Identity)
        # Using register_buffer so they are moved to GPU/CPU with the module 
        # but not treated as learnable parameters for the optimizer.
        #self.register_buffer('R', torch.eye(latent_dim))
        self.register_buffer('Q', torch.eye(latent_dim))
        self.register_buffer('S', torch.zeros(latent_dim,latent_dim))
        self.register_buffer('F', torch.zeros(latent_dim,latent_dim))
        
        # Hyperparameters
        self.eta = eta
        self.reg_c = reg_c # regularization c
        
        self.n_opt_v = n_opt_v
        self.n_opt_pred = n_opt_pred

        # Optimizers
        self.opt_u = torch.optim.SGD(self.encoder_u.parameters(), lr=lr_theta)
        self.opt_v = torch.optim.SGD(self.encoder_v.parameters(), lr=lr_psi)

        self.update_mode_cold_start= update_mode_cold_start
        self.running_grad = None
        self.first_update = True
        self.full_solve_v = full_solve_v


    def forward_backward(self, data_gen):
        """
        One full iteration of SCEgrad using fresh batches.
        """

        # --- UPDATE PSI (encoder_v) ---
        
        raise NotImplementedError



    @torch.no_grad()
    def compute_alignment(self, data_gen, num_batches=10):
        """
        Computes D = SQ and F = E[uu^T].
        Analyzes how well the eigenvectors of F align with the operator D.
        """
        # 1. Compute D = SQ
        # Note: Ensure Q and S are updated before calling this
        D = torch.matmul(self.S, self.Q)
        spectrum_x = compute_spectrum(data_gen, self.encoder_u, view="x", num_batches=num_batches)
        spectrum_y = compute_spectrum(data_gen, self.encoder_v, view="y", num_batches=num_batches)


        R_u = spectrum_x["R_u"]
        Evv = spectrum_y["Euu"]
        
        alignments_D = compute_alignment_func(D,R_u)
        alignments_S = compute_alignment_func(Evv,R_u)


        return {
            "alignments_D_F": alignments_D,       # List of k alignment scores
            "alignments_S_F": alignments_S,
            "eigenvalues_F": spectral_decomposition["S_u"],            # Importance of each direction
            "eigenvalues_S": spectral_decomposition["S_v"],            # Importance of each direction
        }
    def reset_parameters(self, mode='standard', scale=1e-5):
        """
        Custom initialization to test the instability of zero.
        """
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    if mode == 'near_zero':
                        # Critical for your experiment: initialize 
                        # weights extremely close to 0.
                        nn.init.normal_(m.weight, mean=0.0, std=scale)
                    else:
                        # Standard Xavier initialization
                        nn.init.xavier_uniform_(m.weight)
                    
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        # Reset internal buffers Q and S to Identity
        self.Q.copy_(torch.eye(self.k, device=self.Q.device))
        self.S.copy_(torch.zeros(self.k, self.k, device=self.S.device))
        self.F.copy_(torch.zeros(self.k, self.k, device=self.F.device))
        self.first_update = True


class CSEgrad(RepLearner):
    def __init__(self, encoder_u, encoder_v, **kwargs):

        super().__init__(encoder_u, encoder_v, **kwargs)



    
        

    def forward_backward(self, data_gen):
        """
        One full iteration of SCEgrad using fresh batches.
        """

        # --- UPDATE PSI (encoder_v) ---






        if self.full_solve_v:
            self.encoder_v.weight.data=lin_reg_full_solve(data_gen, self.n_opt_v, self.encoder_u)
            loss_reg = 0.*torch.norm(self.encoder_v.weight.data)
        else:
            for n in range(self.n_opt_v):
                batch_reg_x, batch_reg_y = next(data_gen)
                self.opt_v.zero_grad()
                # Compute L_reg (Assuming a regression/reconstruction loss or similar)
                # Since the algorithm LaTeX doesn't define L_reg explicitly, 
                # I'll provide a placeholder matching the gradient requirement.
                with torch.no_grad():
                    u_fixed = self.encoder_u(batch_reg_x).detach()
                v_active = self.encoder_v(batch_reg_y)
                loss_reg = compute_l_reg(u_fixed, v_active)
                loss_reg.backward()
                self.opt_v.step()


        # --- UPDATE Q, S (LinPred) ---
        with torch.no_grad():

            S,F = update_cov(data_gen, self.encoder_u, self.encoder_v, 
                self.S, self.F, self.eta, 
                self.n_opt_pred, 
                cold_start=True,
                symmetric_cov_uv=True,
                EMA=False)
            if self.update_mode_cold_start or self.first_update:
                self.S = S
                self.F = F
                self.first_update = False
            else:
                self.S  = (1-self.eta)*self.S + self.eta*S
                self.F  = (1-self.eta)*self.F + self.eta*F

            # for iter_pred in range(self.n_opt_pred):
            #     batch_pred_x, batch_pred_y = next(data_gen)
            #     self.S = update_cov(self.S, self.eta, 
            #                         batch_pred_y, self.encoder_v)
            #     self.F = update_cov(self.F, self.eta, 
            #                         batch_pred_x, self.encoder_u)

            self.Q = compute_inverse(self.F, self.reg_c)




        # --- UPDATE THETA (encoder_u) ---
        batch_cse_x, batch_cse_y = next(data_gen)
        self.opt_u.zero_grad()
        u_active = self.encoder_u(batch_cse_x)
        with torch.no_grad():
            v_fixed = self.encoder_v(batch_cse_y)
        # Gradient of L_aux with respect to theta
        loss_aux = compute_l_aux(u_active, v_fixed, self.Q, self.S, self.reg_c)
        loss_aux.backward()
        
        #print(lol)
        self.running_grad = compute_avg_field(self.running_grad,self.encoder_u.weight.grad, self.eta)
        grad_norm = torch.norm(self.encoder_u.weight.grad)
        running_grad_norm = torch.norm(self.running_grad)
        self.opt_u.step()

        loss_CSE = compute_l_CSE(self.Q ,self.S, u_active.detach(), self.reg_c)

        return {
            "loss_reg": loss_reg.item(),
            "loss_aux": loss_aux.item(),
            "loss_CSE": loss_CSE.item(),
            "grad_norm": grad_norm.item(),
            "running_grad_norm": running_grad_norm.item(),
            "Q_norm": torch.norm(self.Q).item(),
            "S_norm": torch.norm(self.S).item()
        }


class SemiGradSSL(RepLearner):
    def __init__(self, encoder_u, encoder_v,**kwargs):

        super().__init__(encoder_u, encoder_v, **kwargs)

        self.encoder_v = self.encoder_u

    def forward_backward(self, data_gen):
        """
        One full iteration of SCEgrad using fresh batches.
        """

        # --- UPDATE PSI (encoder_v) ---
        

        # --- UPDATE Q, S (LinPred) ---
        with torch.no_grad():
            # for iter_pred in range(self.n_opt_pred):
            #     batch_pred_x, batch_pred_y = next(data_gen)
            #     self.S = update_cov(self.S, self.eta, 
            #                         batch_pred_x, self.encoder_u,
            #                         batch_pred_y, self.encoder_v)
            #     self.F = update_cov(self.F, self.eta, 
            #                         batch_pred_x, self.encoder_u)

            # self.S, self.F = update_cov(data_gen, self.encoder_u, self.encoder_v, 
            #     self.S,self.F, self.eta, 
            #     self.n_opt_pred, 
            #     cold_start=self.update_mode_cold_start,
            #     symmetric_cov_uv=False,
            #     EMA=self.update_mode_ema)

            S,F = update_cov(data_gen, self.encoder_u, self.encoder_v, 
                            self.S, self.F, self.eta, 
                            self.n_opt_pred, 
                            cold_start=True,
                            symmetric_cov_uv=False,
                            EMA=False)
            if self.update_mode_cold_start or self.first_update:
                self.S = S
                self.F = F
                self.first_update=  False
            else:
                self.S  = (1-self.eta)*self.S + self.eta*S
                self.F  = (1-self.eta)*self.F + self.eta*F

            self.Q = compute_inverse(self.F, self.reg_c)
        # --- UPDATE THETA (encoder_u) ---
        batch_cse_x, batch_cse_y = next(data_gen)
        self.opt_u.zero_grad()
        u_active = self.encoder_u(batch_cse_x)
        with torch.no_grad():
            v_fixed = self.encoder_v(batch_cse_y)
        # Gradient of L_aux with respect to theta
        loss_aux = compute_SG_SSL(u_active, v_fixed, self.Q, self.S, self.reg_c, self.encoder_u)
        loss_aux.backward()
        self.running_grad = compute_avg_field(self.running_grad,self.encoder_u.weight.grad, self.eta)
        grad_norm = torch.norm(self.encoder_u.weight.grad)
        running_grad_norm = torch.norm(self.running_grad)

        self.opt_u.step()


        return {
            "loss_reg": 0.,
            "loss_aux": loss_aux.item(),
            "loss_CSE": 0.,
            "grad_norm": grad_norm.item(),
            "running_grad_norm": running_grad_norm.item(),
            "Q_norm": torch.norm(self.Q).item(),
            "S_norm": torch.norm(self.S).item()
        }






























