import torch
import torch.nn as nn

import torch
import torch.nn as nn

from linear_predictors import (exact_inverse, 
                                newton_update, 
                                newton_cholesky_update, 
                                solve_with_cholesky,
                                PDInverseModule)
from functools import partial
from linear_predictors import update_cov_old
import math

def get_cosine_momentum(epoch, T_max, m_start, m_end):
    return m_end + 0.5 * (m_start - m_end) * (1 + math.cos(math.pi * epoch / T_max))



def compute_l_reg(u, v):
    # Placeholder for L_reg(U, V)
    return 0.5* torch.mean(torch.sum((u-v)**2, dim=1))


    #torch.nn.functional.mse_loss(v, u)

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


def compute_SG_SSL(u,v, P, reg_c,model):
    
    l2_penalty = 0
    for param in model.parameters():
        l2_penalty += torch.sum(param ** 2)
    pred = torch.einsum('rl,bl->br',P,u)
    return 0.5*torch.mean(torch.sum((pred-v)**2,dim=1)) + 0.5*reg_c*l2_penalty


def compute_svd(data_gen, 
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

    batch_size = v.shape[0]
    if u is None:
        u = v
    # 1. Update S: S = (1 - eta_S)S + eta_S * Mean(v v^T)
    if S is None:
        S = torch.matmul(v.t(), u) / batch_size
    else:
        S = (1 - eta) * S + eta * torch.matmul(v.t(), u) / batch_size
    return S






class RepLearner(nn.Module):
    def __init__(self, encoder_u, encoder_v, 
                        latent_dim=10, 
                        eta= 0.1,
                        reg_c = 1.0,
                        lr_u= 0.001,
                        lr_v= 0.001,
                        n_opt_v= 10,
                        n_opt_pred=1,
                        n_accum_pred=10,
                        update_mode_cold_start=False,
                        full_solve_v=False,
                        T_max = 20000,
                        params_Q = {"update_rule":"exact_inverse", 
                                    "momentum": False,
                                    "cold_start": False,
                                    "kwargs":None}):

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
        if params_Q["update_rule"]=="newton_cholesky_update":
            self.register_buffer('D', torch.zeros(latent_dim))
            self.register_buffer('L', torch.zeros(latent_dim, latent_dim))


        # Hyperparameters
        self.eta = eta
        self.reg_c = reg_c # regularization c
        
        self.n_opt_v = n_opt_v
        self.n_opt_pred = n_opt_pred
        self.n_accum_pred= n_accum_pred
        self.T_max = T_max
        self.lr_u= lr_u
        self.lr_v = lr_v
        # Optimizers
        self.opt_u = torch.optim.SGD(self.encoder_u.parameters(), lr=lr_u)
        self.opt_v = torch.optim.SGD(self.encoder_v.parameters(), lr=lr_v)
        self.scheduler_u = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt_u, T_max=T_max, eta_min=0.000001) 
        self.scheduler_v = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt_v, T_max=T_max, eta_min=0.000001) 
        self.scheduler_eta =  partial(get_cosine_momentum, T_max=T_max, 
                                                           m_start=self.eta, 
                                                           m_end=0.001)
        

        self.update_mode_cold_start= update_mode_cold_start
        self.running_grad = None
        self.first_update = True
        self.full_solve_v = full_solve_v
        self.sym_cov = None
        self.params_Q = params_Q
        self.counter = 0
        if params_Q["kwargs"] is None:
            params_Q["kwargs"] = {}


    def compute_target_spectrum(self, data_gen):
        Sx = data_gen.info["S_1"]
        Sy = data_gen.info["S_2"]
        sigma = data_gen.info["sigma"]
        target_spectrum = torch.abs(Sx*Sy)/(torch.sqrt(Sx**2+sigma**2)* torch.sqrt(Sy**2+sigma**2))
        target_spectrum,_ = torch.sort(target_spectrum, descending=True) 
        return target_spectrum



    def update_cov(self,data_gen, sym_cov):
        with torch.no_grad():

            #for iter_pred in range(self.n_opt_pred):
            S = torch.zeros_like(self.S)
            F = torch.zeros_like(self.F)
            for iter_accum in range(self.n_accum_pred):
                batch_pred_x, batch_pred_y = next(data_gen)
                u_code = self.encoder_u(batch_pred_x)
                v_code = self.encoder_v(batch_pred_y)
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

            if self.params_Q["cold_start"]:
                self.Q = torch.eye(self.Q.shape[0], device=self.Q.device, dtype = self.Q.dtype)
            for iter_opt_pred in range(self.n_opt_pred):
                S,F = self.update_cov(data_gen, sym_cov=self.sym_cov)
                if self.params_Q["momentum"]:
                    F = self.F

                self.Q = self.compute_Q(F)



    def compute_Q(self, F):
        if self.params_Q["update_rule"]=="exact_inverse":
            return exact_inverse(F, self.reg_c)
        elif self.params_Q["update_rule"]=="newton_update":
            return newton_update(self.Q, F, self.reg_c, **self.params_Q["kwargs"])
        elif self.params_Q["update_rule"]=="newton_cholesky_update":
            Q_new, self.D, self.L = newton_cholesky_update(self.D,self.L, F, 
                                                            self.reg_c, 
                                                            **self.params_Q["kwargs"])
            return Q_new

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
        spectrum_x = compute_svd(data_gen, self.encoder_u, view="x", num_batches=num_batches)
        spectrum_y = compute_svd(data_gen, self.encoder_v, view="y", num_batches=num_batches)


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
        
    def scheduler_step(self):
        self.scheduler_u.step()
        self.scheduler_v.step()
        self.counter +=1
        self.eta = self.scheduler_eta(self.counter)

class CSEgrad(RepLearner):
    def __init__(self, encoder_u, encoder_v, **kwargs):

        super().__init__(encoder_u, encoder_v, **kwargs)


    def compute_spectrum(self, data_gen):
        svd = compute_svd(data_gen, self.encoder_u, view="x", num_batches=100)

        target_spectrum = torch.zeros_like(torch.zeros_like(svd["S_u"], dtype=torch.float64))
        target_spectrum_tmp = self.compute_target_spectrum(data_gen) 
        target_spectrum_tmp = torch.where(target_spectrum_tmp < self.reg_c, 0.0, target_spectrum_tmp)
        

        n = target_spectrum_tmp.shape[0]
        k = svd["S_u"].shape[0]
        num_to_copy = min(n, k)
        target_spectrum[:num_to_copy] = target_spectrum_tmp[:num_to_copy]


        spectrum = torch.where(target_spectrum > self.reg_c, svd["S_u"] + self.reg_c, svd["S_u"])


        return spectrum, target_spectrum 



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




        # Scheduling

        self.scheduler_step()




        return {
            "loss_reg": loss_reg.item(),
            "loss_aux": loss_aux.item(),
            "loss_CSE": loss_CSE.item(),
            "grad_norm": grad_norm.item(),
            "running_grad_norm": running_grad_norm.item(),
            "Q_norm": torch.norm(self.Q).item(),
            "S_norm": torch.norm(self.S).item()
        }

from implicit_linear_predictor import (linear_probe_loss,
                                        adjoint_pred_loss,
                                        auxiliary_loss,
                                        IterativeLinearSolver,
                                        MomentumRegressor,
                                        IterativeDiffRegressor)



class CSEBilevel(RepLearner):
    def __init__(self, encoder_u, encoder_v, **kwargs):

        super().__init__(encoder_u, encoder_v, **kwargs)
        optimizer_ctor = lambda params: torch.optim.SGD(params, lr=self.eta)

        self.regressor = IterativeDiffRegressor(torch.zeros(self.k, self.k),
                                                torch.zeros(self.k, self.k),
                                                optimizer_ctor= optimizer_ctor,
                                                reg = self.reg_c)
        

        # self.regressor = MomentumRegressor(torch.zeros(self.k, self.k),
        #                                         torch.zeros(self.k, self.k),
        #                                         reg = self.reg_c,
        #                                         eta = self.eta)
        self.rep_loss = auxiliary_loss
        self.inverse = PDInverseModule()

    
    def compute_spectrum(self, data_gen):
        svd_u = compute_svd(data_gen, self.encoder_u, view="x", num_batches=100)
        svd_v = compute_svd(data_gen, self.encoder_v, view="y", num_batches=100)

        _,svd,_ = torch.linalg.svd(svd_u["Euu"]+svd_v["Euu"])


        target_spectrum = torch.zeros_like(torch.zeros_like(svd, dtype=torch.float64))
        target_spectrum_tmp = self.compute_target_spectrum(data_gen) 
        target_spectrum_tmp = torch.where(torch.sqrt(target_spectrum_tmp) < self.reg_c, 0.0, target_spectrum_tmp)
        

        n = target_spectrum_tmp.shape[0]
        k = svd.shape[0]
        num_to_copy = min(n, k)
        target_spectrum[:num_to_copy] = target_spectrum_tmp[:num_to_copy]

        spec = (svd + self.reg_c)**2
        spectrum = torch.where(torch.sqrt(target_spectrum) > self.reg_c, spec, svd)



        return spectrum, target_spectrum 

    def forward_backward(self, data_gen): 
        """
        One full iteration of SCEgrad using fresh batches.
        """

        # --- UPDATE PSI (encoder_v) ---
        for n in range(self.n_opt_pred):
            with torch.no_grad():
                batch_cse_x, batch_cse_y = next(data_gen)
                X = self.encoder_u(batch_cse_x)
                Y = self.encoder_v(batch_cse_y)
            pred, adj_pred = self.regressor(X,Y)


        self.opt_u.zero_grad()
        self.opt_v.zero_grad()
        batch_cse_x, batch_cse_y = next(data_gen)
        X = self.encoder_u(batch_cse_x)
        Y = self.encoder_v(batch_cse_y)        

        loss_aux = self.rep_loss(X,Y, pred, adj_pred, self.reg_c)

        loss = 0.5*torch.trace(pred)-0.5*self.reg_c*(torch.mean(torch.sum(X**2, dim=1)) + torch.mean(torch.sum(Y**2, dim=1)))





        
        # Gradient of L_aux with respect to theta
        loss_aux.backward()

        # grad_U =1.* self.encoder_u.weight.grad
        # grad_V =1.* self.encoder_v.weight.grad
        

        # self.opt_u.zero_grad()
        # self.opt_v.zero_grad()

        # u_active = self.encoder_u(batch_cse_x)
        # v_active = self.encoder_v(batch_cse_y)
        # B = u_active.shape[0]
        # inter_cov = torch.einsum("mk,ml->kl"  ,v_active,u_active)/B
        # cov_u = torch.einsum("mk,ml->kl", u_active,u_active)/B
        # cov_v =torch.einsum("mk,ml->kl", v_active,v_active)/B
        
        # cov = cov_v+ cov_u + self.reg_c*torch.eye(cov_u.shape[0], device=cov_u.device, dtype=cov_u.dtype)
        # inv_cov = self.inverse(cov)
        # loss = -0.5*torch.sum((inter_cov + inter_cov.t())*inv_cov)   
        # loss = loss + 0.5*self.reg_c*(torch.sum(v_active**2) + torch.sum(u_active**2))/B



        # loss.backward()

        # grad_UU =1.* self.encoder_u.weight.grad
        # grad_VV =1.* self.encoder_v.weight.grad        


        # print(f"grad_U err: {torch.norm(grad_UU-grad_U)}")
        # print(f"grad_U err: {torch.norm(grad_VV-grad_V)}")

        # print(lol)


        self.running_grad = compute_avg_field(self.running_grad,self.encoder_u.weight.grad, self.eta)
        grad_norm = torch.norm(self.encoder_u.weight.grad)
        running_grad_norm = torch.norm(self.running_grad)

        self.opt_u.step()
        self.opt_v.step()
        #self.scheduler_step()



        return {
            "loss_reg": loss.item(),
            "loss_aux": loss_aux.item(),
            "loss_CSE": loss.item(),
            "grad_norm": grad_norm.item(),
            "running_grad_norm": running_grad_norm.item(),
            "Q_norm": torch.norm(self.Q).item(),
            "S_norm": torch.norm(self.S).item()
        }







class CSEBilevelJoint(RepLearner):
    def __init__(self, encoder_u, encoder_v, **kwargs):

        super().__init__(encoder_u, encoder_v, **kwargs)
        optimizer_ctor = lambda params: torch.optim.SGD(params, lr=self.eta)

        self.regressor = IterativeDiffRegressor(torch.zeros(self.k, self.k),
                                                torch.zeros(self.k, self.k),
                                                optimizer_ctor= optimizer_ctor,
                                                reg = self.reg_c)
        



        self.rep_loss = auxiliary_loss 
        #self.inverse = PDInverseModule()


    def compute_spectrum(self, data_gen):
        svd_u = compute_svd(data_gen, self.encoder_u, view="x", num_batches=100)
        svd_v = compute_svd(data_gen, self.encoder_v, view="y", num_batches=100)

        _,svd,_ = torch.linalg.svd(svd_u["Euu"]+svd_v["Euu"])


        target_spectrum = torch.zeros_like(torch.zeros_like(svd, dtype=torch.float64))
        target_spectrum_tmp = self.compute_target_spectrum(data_gen) 
        target_spectrum_tmp = torch.where(torch.sqrt(target_spectrum_tmp) < self.reg_c, 0.0, target_spectrum_tmp)
        

        n = target_spectrum_tmp.shape[0]
        k = svd.shape[0]
        num_to_copy = min(n, k)
        target_spectrum[:num_to_copy] = target_spectrum_tmp[:num_to_copy]

        spec = (svd + self.reg_c)**2
        spectrum = torch.where(torch.sqrt(target_spectrum) > self.reg_c, spec, svd)



        return spectrum, target_spectrum 

    def forward_backward(self, data_gen): 
        """
        One full iteration of SCEgrad using fresh batches.
        """



        self.opt_u.zero_grad()
        self.opt_v.zero_grad()
        self.regressor.optimizer.zero_grad()

        batch_cse_x, batch_cse_y = next(data_gen)
        X = self.encoder_u(batch_cse_x)
        Y = self.encoder_v(batch_cse_y)        
        pred = self.regressor.pred
        adjoint = self.regressor.adjoint
        X_det, Y_det = X.detach(), Y.detach()


        pred_loss = self.regressor.pred_loss(X_det, Y_det, pred, self.reg_c)
        adjoint_loss = self.regressor.adjoint_loss(X_det, Y_det, adjoint, self.reg_c)
        loss_aux = self.rep_loss(X,Y, pred.detach(), adjoint.detach(), self.reg_c)
        total_loss = pred_loss + adjoint_loss + loss_aux
        loss = 0.5*torch.trace(pred)-0.5*self.reg_c*(torch.mean(torch.sum(X**2, dim=1)) + torch.mean(torch.sum(Y**2, dim=1)))





        
        # Gradient of L_aux with respect to theta
        total_loss.backward()

        # grad_U =1.* self.encoder_u.weight.grad
        # grad_V =1.* self.encoder_v.weight.grad
        

        # self.opt_u.zero_grad()
        # self.opt_v.zero_grad()

        # u_active = self.encoder_u(batch_cse_x)
        # v_active = self.encoder_v(batch_cse_y)
        # B = u_active.shape[0]
        # inter_cov = torch.einsum("mk,ml->kl"  ,v_active,u_active)/B
        # cov_u = torch.einsum("mk,ml->kl", u_active,u_active)/B
        # cov_v =torch.einsum("mk,ml->kl", v_active,v_active)/B
        
        # cov = cov_v+ cov_u + self.reg_c*torch.eye(cov_u.shape[0], device=cov_u.device, dtype=cov_u.dtype)
        # inv_cov = self.inverse(cov)
        # loss = -0.5*torch.sum((inter_cov + inter_cov.t())*inv_cov)   
        # loss = loss + 0.5*self.reg_c*(torch.sum(v_active**2) + torch.sum(u_active**2))/B



        # loss.backward()

        # grad_UU =1.* self.encoder_u.weight.grad
        # grad_VV =1.* self.encoder_v.weight.grad        


        # print(f"grad_U err: {torch.norm(grad_UU-grad_U)}")
        # print(f"grad_U err: {torch.norm(grad_VV-grad_V)}")

        # print(lol)


        self.running_grad = compute_avg_field(self.running_grad,self.encoder_u.weight.grad, self.eta)
        grad_norm = torch.norm(self.encoder_u.weight.grad)
        running_grad_norm = torch.norm(self.running_grad)

        self.opt_u.step()
        self.opt_v.step()
        self.regressor.optimizer.step()
        #self.scheduler_step()



        return {
            "loss_reg": loss.item(),
            "loss_aux": loss_aux.item(),
            "loss_CSE": loss.item(),
            "grad_norm": grad_norm.item(),
            "running_grad_norm": running_grad_norm.item(),
            "Q_norm": torch.norm(self.Q).item(),
            "S_norm": torch.norm(self.S).item()
        }










class CSEExact(RepLearner):
    def __init__(self, encoder_u, encoder_v, **kwargs):

        super().__init__(encoder_u, encoder_v, **kwargs)

        self.inverse = PDInverseModule()
    # def filter(self, data_gen):
    #     Sx = data_gen.info["S_1"]
    #     Sy = data_gen.info["S_2"]
    #     sigma = data_gen.info["sigma"]
    #     Corr = Sx*Sy/(torch.sqrt(Sx**2+sigma**2)* torch.sqrt(Sy**2+sigma**2))

    #     discriminant = Corr-self.reg_c
    #     return torch.clamp(discriminant, min=0)
    
    def compute_spectrum(self, data_gen):
        svd_u = compute_svd(data_gen, self.encoder_u, view="x", num_batches=100)
        svd_v = compute_svd(data_gen, self.encoder_v, view="y", num_batches=100)

        _,svd,_ = torch.linalg.svd(svd_u["Euu"]+svd_v["Euu"])


        target_spectrum = torch.zeros_like(torch.zeros_like(svd, dtype=torch.float64))
        target_spectrum_tmp = self.compute_target_spectrum(data_gen) 
        target_spectrum_tmp = torch.where(torch.sqrt(target_spectrum_tmp) < self.reg_c, 0.0, target_spectrum_tmp)
        

        n = target_spectrum_tmp.shape[0]
        k = svd.shape[0]
        num_to_copy = min(n, k)
        target_spectrum[:num_to_copy] = target_spectrum_tmp[:num_to_copy]

        spec = (svd + self.reg_c)**2
        spectrum = torch.where(torch.sqrt(target_spectrum) > self.reg_c, spec, svd)



        return spectrum, target_spectrum 

    def forward_backward(self, data_gen): 
        """
        One full iteration of SCEgrad using fresh batches.
        """

        # --- UPDATE PSI (encoder_v) ---
        batch_cse_x, batch_cse_y = next(data_gen)



        self.opt_u.zero_grad()
        self.opt_v.zero_grad()
        u_active = self.encoder_u(batch_cse_x)
        v_active = self.encoder_v(batch_cse_y)
        
        B = u_active.shape[0]
        inter_cov = torch.einsum("mk,ml->kl"  ,v_active,u_active)/B
        cov_u = torch.einsum("mk,ml->kl", u_active,u_active)/B
        cov_v =torch.einsum("mk,ml->kl", v_active,v_active)/B
        
        cov = cov_v+ cov_u + self.reg_c*torch.eye(cov_u.shape[0], device=cov_u.device, dtype=cov_u.dtype)
        inv_cov = self.inverse(cov)
        loss = -0.5*torch.sum( (inter_cov + inter_cov.t())*inv_cov)   
        loss = loss + 0.5*self.reg_c*(torch.sum(v_active**2) + torch.sum(u_active**2))/B
        # Gradient of L_aux with respect to theta
        loss.backward()
        self.running_grad = compute_avg_field(self.running_grad,self.encoder_u.weight.grad, self.eta)
        grad_norm = torch.norm(self.encoder_u.weight.grad)
        running_grad_norm = torch.norm(self.running_grad)

        self.opt_u.step()
        self.opt_v.step()
        #self.scheduler_step()



        return {
            "loss_reg": loss.item(),
            "loss_aux": loss.item(),
            "loss_CSE": loss.item(),
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
        self.register_buffer('P', torch.zeros(self.k,self.k))
        self.P = nn.Linear(self.k, self.k, bias=False)
        self.opt_pred = torch.optim.SGD(self.P.parameters(), lr=self.eta)
        self.scheduler_pred = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt_pred, T_max=self.T_max, eta_min=0.000001) 

    def compute_spectrum(self, data_gen):
        Sx = data_gen.info["S_1"]
        Sy = data_gen.info["S_2"]
        sigma = data_gen.info["sigma"]
        Corr = torch.abs(Sx*Sy)/(Sx**2+sigma**2)
        Corr,_ = torch.sort(Corr, descending=True) 

        thresh = self.reg_c/(Sx**2+sigma**2)
        discriminant = Corr**2 - 4 * thresh
        ratio = torch.sqrt(((Sx**2+sigma**2)/(Sy**2+sigma**2)))

        svd = compute_svd(data_gen, self.encoder_u, view="x", num_batches=100)

        X = 1./(Sx**2+sigma**2)
        n = X.size(0)
        k = svd["S_u"].shape[0]
        num_to_copy = min(n, k)

        cov_X = torch.ones_like(svd["S_u"], dtype = svd["S_u"].dtype)
        cov_X[:num_to_copy] = X[:num_to_copy] 
        
        singular_val = torch.sqrt((svd["S_u"]*cov_X))

        mask = (discriminant >= 0) & (thresh <= 0.25) 

        mask = torch.nn.functional.pad(mask[:k], (0, max(0, k - n)), value=False)

        thresh  = torch.nn.functional.pad(thresh[:k], (0, max(0, k - n)), mode='constant', value=0.0) 
        ratio = torch.nn.functional.pad(ratio[:k], (0, max(0, k - n)), mode='constant', value=1.0) 
        Corr = torch.nn.functional.pad(Corr[:k], (0, max(0, k - n)), mode='constant', value=0.0) 
        target_spectrum = Corr/ratio
        spectrum = torch.where(
            mask,
              ratio*(singular_val + thresh/singular_val),
            svd["S_u"]
        )

        return spectrum, target_spectrum 


    def forward_backward(self, data_gen):
        """
        One full iteration of SCEgrad using fresh batches.
        """

        # --- UPDATE PSI (encoder_v) ---
        

        # --- UPDATE Q, S (LinPred) ---

        #self.update_linear_predictor(data_gen)
        if self.full_solve_v:
           
            with torch.no_grad():
                self.update_cov(data_gen, sym_cov=False)
                predictor = solve_with_cholesky(self.S, self.F, self.reg_c)

        else:
            for pred_steps in range(self.n_opt_pred):
                self.opt_pred.zero_grad()
                with torch.no_grad():
                    batch_cse_x, batch_cse_y = next(data_gen)
                    v_fixed = self.encoder_v(batch_cse_y)
                    u_fixed = self.encoder_u(batch_cse_x)
                loss_aux = compute_SG_SSL(u_fixed, v_fixed, self.P.weight, self.reg_c, self.encoder_u)
                loss_aux = loss_aux + 0.5*self.reg_c*torch.sum(self.P.weight**2)
                loss_aux.backward()
                self.opt_pred.step()
            predictor = self.P.weight.data

        # --- UPDATE THETA (encoder_u) ---
        batch_cse_x, batch_cse_y = next(data_gen)
        self.opt_u.zero_grad()
        u_active = self.encoder_u(batch_cse_x)
        with torch.no_grad():
            v_fixed = self.encoder_v(batch_cse_y)
        # Gradient of L_aux with respect to theta
        loss_aux = compute_SG_SSL(u_active, v_fixed, predictor, self.reg_c, self.encoder_u)
        loss_aux.backward()
        self.running_grad = compute_avg_field(self.running_grad,self.encoder_u.weight.grad, self.eta)
        grad_norm = torch.norm(self.encoder_u.weight.grad)
        running_grad_norm = torch.norm(self.running_grad)

        self.opt_u.step()
        if not self.full_solve_v:
            self.scheduler_pred.step()


        self.scheduler_step()

        return {
            "loss_reg": 0.,
            "loss_aux": loss_aux.item(),
            "loss_CSE": 0.,
            "grad_norm": grad_norm.item(),
            "running_grad_norm": running_grad_norm.item(),
            "Q_norm": torch.norm(self.Q).item(),
            "S_norm": torch.norm(self.S).item()
        }






class UnconstrainedCCA(RepLearner):
    def __init__(self, encoder_u, encoder_v,**kwargs):

        super().__init__(encoder_u, encoder_v, **kwargs)
        #self.opt_u = torch.optim.Adam(self.encoder_u.parameters(), lr=self.lr_u)
        #self.opt_v = torch.optim.Adam(self.encoder_v.parameters(), lr=self.lr_v)


    def compute_spectrum(self, data_gen):
        target_spectrum = self.compute_target_spectrum(data_gen) 
        svd = compute_svd(data_gen, self.encoder_u, view="x", num_batches=100)
        spectrum = 2.* svd["S_u"]
        return spectrum, target_spectrum 


    def forward_backward(self, data_gen):
        """
        One full iteration of SCEgrad using fresh batches.
        """

        # --- UPDATE PSI (encoder_v) ---
        

        # --- UPDATE Q, S (LinPred) ---

        #self.update_linear_predictor(data_gen)


        # --- UPDATE THETA (encoder_u) ---
        batch_cse_x, batch_cse_y = next(data_gen)



        self.opt_u.zero_grad()
        self.opt_v.zero_grad()
        u_active = self.encoder_u(batch_cse_x)
        v_active = self.encoder_v(batch_cse_y)
        
        B = u_active.shape[0]
        inter_cov = torch.einsum("mk,mk->"  ,u_active,v_active)/B
        cov_u = torch.einsum("mk,ml->kl", u_active,u_active)/B
        cov_v =torch.einsum("mk,ml->kl", v_active,v_active)/B
        intra_cov = torch.sum((cov_u+cov_v)**2)


        loss = - 4. *inter_cov +  intra_cov
        # Gradient of L_aux with respect to theta
        loss.backward()
        self.running_grad = compute_avg_field(self.running_grad,self.encoder_u.weight.grad, self.eta)
        grad_norm = torch.norm(self.encoder_u.weight.grad)
        running_grad_norm = torch.norm(self.running_grad)

        self.opt_u.step()
        self.opt_v.step()
        self.scheduler_step()

        return {
            "loss_reg": 0.,
            "loss_aux": loss.item(),
            "loss_CSE": loss.item(),
            "grad_norm": grad_norm.item(),
            "running_grad_norm": running_grad_norm.item(),
            "Q_norm": torch.norm(self.Q).item(),
            "S_norm": torch.norm(self.S).item()
        }










  


















