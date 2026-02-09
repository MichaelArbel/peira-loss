import torch
import torch.nn as nn

from utils import instantiate, set_seed
from evaluation import compute_alignment
from experiment_logging import (
    log_alignment_artifacts,
    log_data_info,
    log_epoch_metrics,
    log_target_spectrum,
    log_vector_field_artifacts,
    log_vector_field_metrics,
)
import numpy as np
import time

# 3. The Trainer Class
class CCAExperimentRunner:
    def __init__(self, cfg, logger):
        
        self.logger = logger
        self.device = cfg.device 
        self.dtype = torch.float64 if cfg.type=="double" else torch.float32

        # 1. Construct Infinite Generator
        self.loader = instantiate(cfg.data.name)(device=self.device, dtype=self.dtype, **cfg.data.kwargs)

        self.view1 = nn.Linear(cfg.data.kwargs.d1, cfg.learner.kwargs.latent_dim, bias=False)
        self.view2 = nn.Linear(cfg.data.kwargs.d2, cfg.learner.kwargs.latent_dim, bias=False)

      # Algorithmic Module
        learner_kwargs = dict(cfg.learner.kwargs)
        algorithm_name = learner_kwargs.pop("algorithm_name")
        self.rep_learner = instantiate(cfg.learner.name)(
            self.view1,
            self.view2,
            algorithm_name=algorithm_name,
            t_max=cfg.task.epochs,
            **learner_kwargs,
        )


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
                log_epoch_metrics(self.logger, metrics, epoch)
                
            alignment_dict = self.rep_learner.compute_alignment(self.loader)
            # Save artifacts at the end

            log_alignment_artifacts(self.logger, alignment_dict, epoch)

    def run_vector_field_experiment(self):
        
        set_seed(self.cfg.init_seed)
        # 1. Start very close to zero with a slight random perturbation
        self.rep_learner.reset_parameters(mode=self.cfg.init_mode, scale=self.cfg.init_scale) 
        
        info = {
                k: (v.cpu().numpy() if torch.is_tensor(v) else v) 
                for k, v in self.loader.info.items()
            }
        log_data_info(self.logger, info)
        _, target_spectrum = self.rep_learner.compute_spectrum(self.loader)
        log_target_spectrum(self.logger, target_spectrum.cpu().detach())
        avg_time = 0.
        for step in range(self.cfg.steps_per_epoch):
            
            #print([param for param in self.rep_learner.parameters()])
            with torch.no_grad():
                #spectrum = compute_spectrum(self.loader, self.rep_learner.encoder_u, view="x", num_batches=100)
                spectrum, target_spectrum = self.rep_learner.compute_spectrum(self.loader)

                W = self.rep_learner.encoder_u.weight.t()@self.rep_learner.encoder_u.weight

                alignment = compute_alignment(W, self.loader.info["P"])
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
            log_vector_field_metrics(self.logger, metrics, step, s1, s2, ts1, ts2, align1, align2, avg_time)

            if np.mod(step,100)==0:
                log_vector_field_artifacts(self.logger, alignment, spectrum, step)

            if np.mod(step,10)==0:
                print(f"step: {step}")
                print(f"spectrum: {(s1.item(),s2.item())}")
                print(f"target spectrum: {(ts1.item(),ts2.item())}")
                print(f"alignment: {(align1.item(), align2.item())}")
                print(f"vector field norm: {metrics['running_grad_norm']}")
                print(f"objective: {metrics['loss_CSE']}")
                print(f"avg time: {avg_time}")
