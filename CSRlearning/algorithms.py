import torch
import torch.nn as nn

import torch
import torch.nn as nn

from linear_predictors import (exact_inverse, newton_update)


        



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
    return 0.5*torch.mean(torch.sum((pred-v)**2,dim=1)) + 0.5*reg_c*l2_penalty


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






def _update_cov(S, eta, v, u = None):

    with torch.no_grad():
        batch_size = v.shape[0]
        if u is None:
            u = v
        # 1. Update S: S = (1 - eta_S)S + eta_S * Mean(v v^T)
        S = (1 - eta) * S + eta * torch.matmul(v.t(), u) / batch_size
        return S






class RepLearner(nn.Module):
    def __init__(self, encoder_u, encoder_v, 
                        latent_dim=10, 
                        eta= 0.1,
                        reg_c = 1.0,
                        lr_theta= 0.001,
                        lr_psi= 0.001,
                        n_opt_v= 10,
                        n_opt_pred=1,
                        n_accum_pred=10,
                        update_mode_cold_start=False,
                        full_solve_v=False,
                        params_Q = {"update_rule":"exact_inverse", 
                                    "momentum": False,
                                    "kwargs":{}}):

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
        self.n_accum_pred= n_accum_pred

        # Optimizers
        self.opt_u = torch.optim.SGD(self.encoder_u.parameters(), lr=lr_theta)
        self.opt_v = torch.optim.SGD(self.encoder_v.parameters(), lr=lr_psi)

        self.update_mode_cold_start= update_mode_cold_start
        self.running_grad = None
        self.first_update = True
        self.full_solve_v = full_solve_v
        self.sym_cov = None
        self.params_Q = params_Q

    def update_cov(self,data_gen, sym_cov):
        with torch.no_grad():

            #for iter_pred in range(self.n_opt_pred):
            S = torch.zeros_like(self.S)
            F = torch.zeros_like(self.F)
            for iter_accum in range(self.n_accum_pred):
                batch_pred_x, batch_pred_y = next(data_gen)
                u_code = self.encoder_u(batch_pred_x)
                v_code = self.encoder_u(batch_pred_y)
                eta = 1./(iter_accum+1)
                if sym_cov:
                    S = _update_cov(S, eta, v_code)
                else:
                    S = _update_cov(S, eta, v_code, u_code)
                F = _update_cov(F, eta, u_code)


            if not self.update_mode_cold_start or self.first_update:
                self.S = S
                self.F = F
                self.first_update = False
            else:
                self.S  = (1-self.eta)*self.S + self.eta*S
                self.F  = (1-self.eta)*self.F + self.eta*F

        return S, F
            
    def update_linear_predictor(self,data_gen):

        with torch.no_grad():
            for iter_opt_pred in range(self.n_opt_pred):
                S,F = self.update_cov(data_gen, sym_cov=self.sym_cov)
                if self.params_Q["momentum"]:
                    F = self.F
                self.update_Q(F)


    def update_Q(self, F):
        if self.params_Q["update_rule"]=="exact_inverse":
            self.Q = exact_inverse(F, self.reg_c)
        elif self.params_Q["update_rule"]=="newton_update":
            self.Q = newton_update(self.Q, F, self.reg_c, **self.params_Q["kwargs"])


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
                    elif mode == 'orthogonal':
                        # Standard Xavier initialization
                        nn.init.normal_(m.weight, mean=0.0, std=1.)
                        m.weight.data, _ = torch.linalg.qr(m.weight.data, mode='reduced')
                    
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        # Reset internal buffers Q and S to Identity
        self.Q.copy_(torch.eye(self.k, device=self.Q.device))
        self.S.copy_(torch.zeros(self.k, self.k, device=self.S.device))
        self.F.copy_(torch.zeros(self.k, self.k, device=self.F.device))
        self.first_update = True
        self.sym_cov =True

class CSEgrad(RepLearner):
    def __init__(self, encoder_u, encoder_v, **kwargs):

        super().__init__(encoder_u, encoder_v, **kwargs)


    def filter(self, data_gen):
        Sx = data_gen.info["S_1"]
        Sy = data_gen.info["S_2"]
        sigma = data_gen.info["sigma"]
        Corr = Sx*Sy/(torch.sqrt(Sx**2+sigma**2)* torch.sqrt(Sy**2+sigma**2))

        discriminant = Corr-self.reg_c
        return torch.clamp(discriminant, min=0)
    



    def update_encoder_v(self, data_gen):

        if self.full_solve_v:
            with torch.no_grad():
                for iter_pred in range(self.n_opt_v):
                    batch_pred_x, batch_pred_y = next(data_gen)
                    eta = 1./(iter_pred+1)
                    if iter_pred==0:
                        u = self.encoder_u(batch_pred_x)
                        k, d = u.shape[1], batch_pred_y.shape[1]
                        S = torch.zeros(k,d, device=u.device)
                        F = torch.zeros(d,d, device=u.device)


                    S = _update_cov(S, eta, 
                                    u,batch_pred_y)
                    F = _update_cov(F, eta, 
                                    batch_pred_y)
            
                self.encoder_v.weight.data= S@ exact_inverse(F,0.)            

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
        return loss_reg

    def update_encoder_u(self, data_gen):

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

        return loss_aux, grad_norm, running_grad_norm, u_active

    def forward_backward(self, data_gen):
        """
        One full iteration of SCEgrad using fresh batches.
        """

        # --- UPDATE PSI (encoder_v) ---


        loss_reg = self.update_encoder_v(data_gen)


        # --- UPDATE Q, S (LinPred) ---

        self.update_linear_predictor(data_gen)



        # --- UPDATE THETA (encoder_u) ---
        loss_aux, grad_norm, running_grad_norm, u_active = self.update_encoder_u(data_gen)

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
        self.sym_cov =False

    def filter(self, data_gen):
        Sx = data_gen.info["S_1"]
        Sy = data_gen.info["S_2"]
        sigma = data_gen.info["sigma"]
        Corr = Sx*Sy/(Sx**2+sigma**2)
        thresh = self.reg_c/(Sx**2+sigma**2)

        discriminant = Corr**2 - 4 * thresh
        mask = (discriminant >= 0) & (thresh <= 0.25)
        val = torch.where(
            mask,
            0.5 * (Corr + torch.sqrt(torch.clamp(discriminant, min=0))),
            torch.zeros_like(Corr, dtype=torch.float64)
        )
        return val**2*(Sx**2+sigma**2)




    def forward_backward(self, data_gen):
        """
        One full iteration of SCEgrad using fresh batches.
        """

        # --- UPDATE PSI (encoder_v) ---
        

        # --- UPDATE Q, S (LinPred) ---

        self.update_linear_predictor(data_gen)
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






























