import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from evaluation import compute_alignment, compute_covariance
from experiment_logging import (
    log_alignment_artifacts,
    log_data_info,
    log_epoch_metrics,
    log_target_spectrum,
    log_vector_field_artifacts,
    log_vector_field_metrics,
)
from utils import instantiate, set_seed, compute_running_average, encode
from pprint import pprint


class CCAExperimentRunner:
    def __init__(self, cfg, logger):
        self.logger = logger
        self.tracker = Tracker()
        self.device = cfg.device
        self.dtype = torch.float64 if cfg.type == "double" else torch.float32

        self.loader = instantiate(cfg.data.name)(
            device=self.device, dtype=self.dtype, **cfg.data.kwargs
        )
 

        algorithm_cfg = cfg.algorithm
        state_cfg = algorithm_cfg.kwargs.state
        regressor_cfg = algorithm_cfg.kwargs.regressor

        d1 = cfg.data.kwargs.d1
        d2 = cfg.data.kwargs.d2
        latent_dim = state_cfg.get("latent_dim")
        self.tied_encoders = bool(state_cfg.get("tied_encoders", False))

        view1 = nn.Linear(d1, latent_dim, bias=False)
        if self.tied_encoders:
            view2 = view1
        else:
            view2 = nn.Linear(d2, latent_dim, bias=False)

        self.encoders, self.optimizers, self.schedulers = build_optimizers_and_schedulers([
                                                                    view1,view2], 
                                                                    self.tied_encoders,
                                                                    cfg.optimizer,
                                                                    cfg.scheduler)


        self.algorithm = instantiate(cfg.algorithm.name)(cfg.algorithm.kwargs)
        self.algorithm.configure(self.encoders, self.optimizers, self.schedulers)

        def cast(module, cast_type):
            if cast_type=="double":
                return module.double()
            elif cast_type=="float":
                return module.float()

        self.encoders = [cast(encoder,cfg.type).to(self.device) for encoder in self.encoders]
        self.algorithm = cast(self.algorithm,cfg.type).to(self.device)


        self.cfg = cfg = cfg.task

    def reset_parameters(self, mode="standard", scale=1e-5):
        with torch.no_grad():
            for m in self.encoders:
                if isinstance(m, nn.Linear):
                    if mode == "near_zero":
                        nn.init.normal_(m.weight, mean=0.0, std=scale)
                    elif mode == "orthogonal":
                        nn.init.normal_(m.weight, mean=0.0, std=1.0)
                        m.weight.data, _ = torch.linalg.qr(m.weight.data, mode="reduced")


    def run(self):
        if self.cfg.name == "train":
            self.train()
        elif self.cfg.name == "vectorfield":
            self.run_vector_field_experiment()
        else:
            raise NotImplementedError

    def  step(self):

            self.algorithm.update_internal_state(self.loader)
            for opt in self.optimizers:
                opt.zero_grad()
            data = next(self.loader)
            output = encode(data, self.encoders, self.algorithm.stop_grad)
            loss = self.algorithm.compute_loss(*output)
            loss.backward()


            for opt in self.optimizers:
                opt.step()
            for scheduler in self.schedulers:
                scheduler.step()
            return loss

    def train(self):
        for epoch in range(self.cfg.epochs):
            print(f"epoch: {epoch}")
            for step in range(self.cfg.steps_per_epoch):
                start= time.time()
                loss = self.step()
                end = time.time()
                metrics, _ = self.tracker.track(loss, self.encoders[0].weight.grad, step, start, end)
                self.logger.log_metrics({**metrics,
                                    "epoch": epoch,
                                    }, log_name="metrics")                


    def run_vector_field_experiment(self):
        set_seed(self.cfg.init_seed)
        self.reset_parameters(mode=self.cfg.init_mode, scale=self.cfg.init_scale)
        info = {k: (v.cpu().numpy() if torch.is_tensor(v) else v)
                    for k, v in self.loader.info.items()
                }
        self.logger.log_artifacts(info, "data_info", artifact_type="pickle")
        for step in range(self.cfg.steps_per_epoch):
            with torch.no_grad():
                spectral_metrics=  self.compute_spectral_metrics()
            start = time.time()
            loss = self.step()
            end = time.time()
            metrics, artifact_dict = self.tracker.track(loss, 
                                         self.encoders[0].weight.grad, 
                                         step,
                                         start,end,
                                         spectral_metrics=spectral_metrics)
            self.logger.log_metrics( metrics, log_name="metrics")

            if np.mod(step, 100) == 0:
                self.logger.log_artifacts(artifact_dict, f"alignments_{step}", artifact_type="pickle")

            if np.mod(step, 10) == 0:
                pprint(metrics)

    def compute_spectral_metrics(self):
        metrics = {}

        inter_cov, cov_lists = compute_covariance(self.encoders, 
                                                  self.loader)

        intra_cov = cov_lists[0]

        if not self.tied_encoders:
             intra_cov += cov_lists[1]
             inter_cov += inter_cov.t()



        R, S, _ = torch.linalg.svd(intra_cov)
        target_spectrum_info = self.loader.info
        metrics["spectrum"], metrics["target_spectrum"] = self.algorithm.compute_spectrum(S, target_spectrum_info)
        
        ## Local alignment
        metrics["local_alignment"] = compute_alignment(inter_cov, R)


        # Target alignment of the weights

        W = self.encoders[0].weight

        if self.tied_encoders:
            W = self.encoders[0].weight
            B = target_spectrum_info["cov_1"] 
            U = target_spectrum_info["singular_vectors_1"] 
        else:
            W = torch.cat([self.encoders[0].weight, self.encoders[1].weight], dim=0)
            B = torch.cat([target_spectrum_info["cov_1"] , target_spectrum_info["cov_2"] ], dim=0)
            U = torch.cat([target_spectrum_info["singular_vectors_1"] ,target_spectrum_info["singular_vectors_2"] ], dim=0)
        
        eigvals, eigvecs = torch.linalg.eigh(B)
        sqrt_B = eigvecs @ torch.diag(torch.sqrt(eigvals.clamp(min=0))) @ eigvecs.T

        M = sqrt_B@W.t()@W@sqrt_B
        metrics["alignment"] = compute_alignment(M,U)
        return metrics

def build_optimizers_and_schedulers( 
        encoders,
        tied_encoders,
        optimizer_spec,
        scheduler_spec):
    
    first_encoder = encoders[0]
    if tied_encoders:
        for i in range(len(encoders)): 
            encoders[i] = first_encoder


    optimizer_ctor = lambda params: instantiate(optimizer_spec.name)(params, **optimizer_spec.kwargs)

    scheduler_ctor = lambda optimizer: instantiate(scheduler_spec.name)(
        optimizer, **scheduler_spec.kwargs)

    if not tied_encoders:
        optimizers = [optimizer_ctor(encoder.parameters()) for encoder in encoders]
    else:
         optimizers = [optimizer_ctor(first_encoder.parameters())]

    schedulers = [scheduler_ctor(opt) for opt in optimizers]

    return encoders, optimizers, schedulers




class Tracker:
    def __init__(self, momentum=0.01):
        self.momentum = 0.01
        self.running_grad = None
        self.avg_time = 0.
    def track(self,  loss, 
                     grad, 
                     step,
                     start, end,
                     spectral_metrics=None):
        self.avg_time = (1 - 1.0 / (step + 1)) * self.avg_time + (1.0 / (step + 1)) * (
                end - start
            )
        grad_norm, running_grad_norm = self.compute_vector_field_norm(grad)
        
        if spectral_metrics:
            spectral_metrics, artifact_dict = self.track_spectral_metrics(spectral_metrics)
            artifact_dict["iter"] =  step
        else:
            spectral_metrics = {}
            artifact_dict = {}

        metrics =  {
                    "loss":loss.item(),
                    "grad_norm": grad_norm.item(),
                    "running_grad_norm": running_grad_norm.item(),
                    "iter": step,
                    "avg_time": self.avg_time,
                    **spectral_metrics,
                }


        return metrics, artifact_dict

    def track_spectral_metrics(self,metrics):
        s1, s2 = metrics['spectrum'][0], metrics['spectrum'][1]
        ts1, ts2 = metrics['target_spectrum'][0], metrics['target_spectrum'][1]
        align1, align2 = metrics["alignment"][0], metrics["alignment"][1]

        spectral_metrics = {"singval_1": s1.item(),
                            "singval_2": s2.item(),
                            "t_singval_1": ts1.item(),
                            "t_singval_2": ts2.item(),
                            "align_1": align1.item(),
                            "align_2": align2.item(),
                            }
        artifact_dict = {
                        "alignment": metrics["alignment"].cpu().numpy(),
                        "spectrum": metrics['spectrum'].cpu().numpy(),
                        }
        return spectral_metrics, artifact_dict

    def  compute_vector_field_norm(self, grad):

        self.running_grad = compute_running_average(
            self.running_grad, grad, self.momentum
        )
        grad_norm = torch.norm(grad)
        running_grad_norm = torch.norm(self.running_grad)
        return grad_norm, running_grad_norm
