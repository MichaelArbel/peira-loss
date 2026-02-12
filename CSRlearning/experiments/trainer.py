import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from experiments.evaluation import compute_alignment
from experiments.factory import instantiate, build_optimizer_bundle
from experiments.pipeline import encode
from experiments.evaluation import compute_spectral_metrics
from experiments.utils import set_seed, compute_running_average
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
        latent_dim = algo_kwargs.get("latent_dim")
        self.tied_encoders = bool(algo_kwargs.get("tied_encoders", False))

        view1 = nn.Linear(d1, latent_dim, bias=False)
        if self.tied_encoders:
            view2 = view1
        else:
            view2 = nn.Linear(d2, latent_dim, bias=False)

        self.encoders, self.optimizers = build_optimizer_bundle(
            [view1, view2],
            self.tied_encoders,
            cfg.optimizer,
            cfg.scheduler,
        )


        self.algorithm = instantiate(cfg.algorithm.name)(**cfg.algorithm.kwargs)
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
        for step in range(self.cfg.steps_per_epoch):
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
            metrics, artifact_dict = self.tracker.track(
                loss,
                self.encoders[0].weight.grad,
                step,
                start,
                end,
                spectral_metrics=spectral_metrics,
            )
            self.logger.log_metrics( metrics, log_name="metrics")

            if spectral_metrics is not None and np.mod(step, 100) == 0:
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
        s1, s2 = metrics.spectrum[0], metrics.spectrum[1]
        ts1, ts2 = metrics.target_spectrum[0], metrics.target_spectrum[1]
        align1, align2 = metrics.alignment[0], metrics.alignment[1]

        spectral_metrics = {"singval_1": s1.item(),
                            "singval_2": s2.item(),
                            "t_singval_1": ts1.item(),
                            "t_singval_2": ts2.item(),
                            "align_1": align1.item(),
                            "align_2": align2.item(),
                            }
        artifact_dict = {
                        "alignment": metrics.alignment.cpu().numpy(),
                        "spectrum": metrics.spectrum.cpu().numpy(),
                        }
        return spectral_metrics, artifact_dict

    def  compute_vector_field_norm(self, grad):

        self.running_grad = compute_running_average(
            self.running_grad, grad, self.momentum
        )
        grad_norm = torch.norm(grad)
        running_grad_norm = torch.norm(self.running_grad)
        return grad_norm, running_grad_norm
