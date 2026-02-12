import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from csrlearning_experiments.evaluation.evaluation import compute_alignment
from csrlearning_experiments.factory import instantiate, build_optimizer_bundle
from csrlearning_experiments.training.pipeline import encode
from csrlearning_experiments.evaluation.evaluation import compute_spectral_metrics
from csrlearning_experiments.utils.utils import set_seed, compute_running_average
from pprint import pprint


class InfiniteLoader:
    def __init__(self, loader):
        self.loader = loader
        self.iterator = iter(loader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


class CCAExperimentRunner:
    def __init__(self, cfg, logger):
        self.logger = logger
        self.tracker = Tracker()
        self.device = cfg.device
        self.dtype = torch.float64 if cfg.type == "double" else torch.float32

        self.loader = instantiate(cfg.data.name)(
            device=self.device, dtype=self.dtype, **cfg.data.kwargs
        )
        self.infinite_loader = InfiniteLoader(self.loader)
 

        algorithm_cfg = cfg.algorithm
        algo_kwargs = algorithm_cfg.kwargs

        d1 = cfg.data.kwargs.d1
        d2 = cfg.data.kwargs.d2
        latent_dim = cfg.latent_dim
        if "latent_dim" in cfg.algorithm.kwargs:
            cfg.algorithm.kwargs.latent_dim = latent_dim 
        self.algorithm = instantiate(cfg.algorithm.name)(**cfg.algorithm.kwargs)
        self.tied_encoders = self.algorithm.tied_encoders

        view1 = instantiate(cfg.model.name)(d1,latent_dim, **cfg.model.kwargs)
        #nn.Linear(d1, latent_dim, bias=False)
        if self.tied_encoders:
            view2 = view1
        else:
            view2 = instantiate(cfg.model.name)(d2,latent_dim, **cfg.model.kwargs)

        self.encoders, self.optimizers = build_optimizer_bundle(
            [view1, view2],
            self.tied_encoders,
            cfg.optimizer,
            cfg.scheduler,
        )


        ext_opts, ext_scheds = self.algorithm.configure(self.encoders)
        for opt, sch in zip(ext_opts, ext_scheds):
            self.optimizers.add(opt, sch, managed_by_algorithm=False)

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

            state = self.algorithm.get_internal_state()
            for _ in range(self.algorithm.inner_steps()):
                update_batch = next(self.infinite_loader)
                with torch.no_grad():
                    update_encoded = encode(update_batch, self.encoders, self.algorithm.stop_grad)
                state = self.algorithm.update_internal_state(update_encoded.x1, update_encoded.x2)
            self.optimizers.zero_grad()
            
            data = next(self.infinite_loader)
            output = encode(data, self.encoders, state.stop_grad)
            loss = self.algorithm(output.x1, output.x2, state)
            loss.backward()

            self.optimizers.step()
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
        if self.loader.info is not None:
            info = {
                k: (v.cpu().numpy() if torch.is_tensor(v) else v)
                for k, v in self.loader.info.items()
            }
            self.logger.log_artifacts(info, "data_info", artifact_type="pickle")
        for step in range(self.cfg.total_steps):
            with torch.no_grad():
                spectral_metrics = compute_spectral_metrics(
                    self.encoders,
                    self.loader,
                    self.algorithm,
                    self.tied_encoders,
                )
            start = time.time()
            loss = self.step()
            end = time.time()

            store_artifacts = np.mod(step, 100) == 0

            metrics, artifact_dict = self.tracker.track(
                loss,
                self.encoders[0].weight.grad,
                step,
                start,
                end,
                spectral_metrics=spectral_metrics,
                with_artifacts = store_artifacts
            )
            self.logger.log_metrics( metrics, log_name="metrics")

            if artifact_dict is not None and store_artifacts:
                self.logger.log_artifacts(artifact_dict, f"alignments_{step}", artifact_type="pickle")

            if np.mod(step, 10) == 0:
                pprint(metrics)





class Tracker:
    def __init__(self, momentum=0.01):
        self.momentum = 0.01
        self.running_grad = None
        self.avg_time = 0.
    def track(self,  loss, 
                     grad, 
                     step,
                     start, end,
                     spectral_metrics=None,
                     with_artifacts=False):
        self.avg_time = (1 - 1.0 / (step + 1)) * self.avg_time + (1.0 / (step + 1)) * (
                end - start
            )
        grad_norm, running_grad_norm = self.compute_vector_field_norm(grad)
        
        if spectral_metrics:
            spectral_metrics, artifact_dict = self.track_spectral_metrics(spectral_metrics, step,
                                                                        with_artifacts=with_artifacts)
            
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

    def track_spectral_metrics(self,metrics, step, with_artifacts=False):
        spectral_metrics = {"singval": [metrics.spectrum[0].item(), metrics.spectrum[1].item()],
                            "target_singval": [metrics.target_spectrum[0].item(), metrics.target_spectrum[1].item()],
                            "align": [metrics.alignment[0].item(), metrics.alignment[1].item()],
                            "local_align": [metrics.local_alignment[0].item(), metrics.local_alignment[1].item()],
                            }
        if with_artifacts:
            artifact_dict = {
                            "alignment": metrics.alignment.cpu().numpy(),
                            "spectrum": metrics.spectrum.cpu().numpy(),
                            "local_aligment": metrics.local_alignment.cpu().numpy(),
                            "iter" : step
                            }
            return spectral_metrics, artifact_dict
        return spectral_metrics, None

    def  compute_vector_field_norm(self, grad):

        self.running_grad = compute_running_average(
            self.running_grad, grad, self.momentum
        )
        grad_norm = torch.norm(grad)
        running_grad_norm = torch.norm(self.running_grad)
        return grad_norm, running_grad_norm
