import torch
import torch.nn as nn

import torch
from utils import instantiate, set_seed
from algorithms import CSEgrad, compute_alignment_func
import numpy as np
import time

# 3. The Trainer Class
class CCATrainer:
    def __init__(self, cfg, logger):
        
        self.logger = logger
        self.device = cfg.device 
        self.dtype = torch.float64 if cfg.type=="double" else torch.float32

        # 1. Construct Infinite Generator
        self.loader = instantiate(cfg.data.name)(device=self.device, dtype=self.dtype, **cfg.data.kwargs)

        self.view1 = nn.Linear(cfg.data.kwargs.d1, cfg.learner.kwargs.latent_dim, bias=False)
        self.view2 = nn.Linear(cfg.data.kwargs.d2, cfg.learner.kwargs.latent_dim, bias=False)

      # Algorithmic Module
        self.rep_learner = instantiate(cfg.learner.name)(self.view1, 
                                                         self.view2, 
                                                         T_max=cfg.task.epochs,
                                                         **cfg.learner.kwargs)


        self.rep_learner = self.rep_learner.to(self.device)
        if cfg.type=="double":
            self.rep_learner = self.rep_learner.double()
        self.cfg = cfg = cfg.task
    def run(self):
        if self.cfg.name=="train":
            self.train()
        elif self.cfg.name=="vectorfield":
            self.run_vector_field_experiment()
        else:
            raise NotImplementedError

    def train(self):
        for epoch in range(self.cfg.epochs):
            print(f"epoch: {epoch}")
            total_loss = 0
            for step in range(self.cfg.steps_per_epoch):
                # Pass the generator itself to allow fresh sampling inside
                metrics = self.rep_learner.forward_backward(self.loader)
                



                # Log metrics for every round
                self.logger.log_metrics({
                    **metrics,
                    "epoch": epoch,
                }, log_name= "metrics")
                
            alignment_dict = self.rep_learner.compute_alignment(self.loader)
            # Save artifacts at the end

            self.logger.log_artifacts({**alignment_dict,
                                       "epoch":epoch}, f"alignment_{epoch}",
                                      artifact_type= "pickle")

    def run_vector_field_experiment(self):
        
        set_seed(self.cfg.init_seed)
        # 1. Start very close to zero with a slight random perturbation
        self.rep_learner.reset_parameters(mode=self.cfg.init_mode, scale=self.cfg.init_scale) 
        
        info = {
                k: (v.cpu().numpy() if torch.is_tensor(v) else v) 
                for k, v in self.loader.info.items()
            }
        self.logger.log_artifacts(info, f"data_info",
                                  artifact_type= "pickle")
        _, target_spectrum = self.rep_learner.compute_spectrum(self.loader)
        self.logger.log_artifacts(target_spectrum.cpu().detach(), f"target_spectrum",
                                  artifact_type= "pickle")
        avg_time = 0.
        for step in range(self.cfg.steps_per_epoch):
            
            #print([param for param in self.rep_learner.parameters()])
            with torch.no_grad():
                #spectrum = compute_spectrum(self.loader, self.rep_learner.encoder_u, view="x", num_batches=100)
                spectrum, target_spectrum = self.rep_learner.compute_spectrum(self.loader)

                W = self.rep_learner.encoder_u.weight.t()@self.rep_learner.encoder_u.weight

                alignment = compute_alignment_func(W, self.loader.info["P"])
            s1, s2 = spectrum[0],spectrum[1]
            ts1, ts2 = target_spectrum[0], target_spectrum[1]
            align1, align2 = alignment[0], alignment[1]
            #print(lol)
            # Perform one SCEgrad update
            start = time.time()
            metrics = self.rep_learner.forward_backward(self.loader)
            end = time.time()
            avg_time = (1-1./(step+1)) * avg_time + (1./(step+1))*(time.time()- start)
            #if np.mod(step,10)==0:
            self.logger.log_metrics({
                    **metrics,
                    "iter": step,
                    "singval_1": s1.item(),
                    "singval_2": s2.item(),
                    "t_singval_1": ts1.item(),
                    "t_singval_2": ts2.item(),
                    "align_1": align1.item(),
                    "align_2": align2.item(),
                    "avg_time": avg_time
                }, log_name= "metrics")

            if np.mod(step,100)==0:
                self.logger.log_artifacts({"alignment": alignment.cpu().numpy(), 
                                           "spectrum":spectrum.cpu().numpy(), 
                                           "iter": step}, 
                                           f"alignments_{step}",
                                            artifact_type= "pickle")

            if np.mod(step,10)==0:
                print(f"step: {step}")
                print(f"spectrum: {(s1.item(),s2.item())}")
                print(f"target spectrum: {(ts1.item(),ts2.item())}")
                print(f"alignment: {(align1.item(), align2.item())}")
                print(f"vector field norm: {metrics["running_grad_norm"]}")
                print(f"objective: {metrics["loss_CSE"]}")
                print(f"avg time: {avg_time}")



