"""Deterministic linear-encoder training for exact PEIRA and Self-distillation."""

from pprint import pprint
import torch
from torch import nn

from peira_learning.algorithms import ExactPEIRA, ExactSelfDistillation
from peira_learning_experiments.factory import instantiate
from peira_learning_experiments.training.pipeline import encode
from peira_learning_experiments.evaluation.evaluation import compute_spectral_metrics
from peira_learning_experiments.utils.utils import set_seed


class CCAExperimentRunner:
    def __init__(self, cfg, logger):
        self.logger = logger
        self.cfg = cfg
        dtype = torch.float64 if cfg.type == "double" else torch.float32
        self.loader = instantiate(cfg.data.name)(
            device=cfg.device, dtype=dtype, **cfg.data.kwargs
        )
        self.algorithm = instantiate(cfg.algorithm.name)(**cfg.algorithm.kwargs)
        if not isinstance(self.algorithm, (ExactPEIRA, ExactSelfDistillation)):
            raise ValueError("Only exact PEIRA and Self-distillation are supported.")
        self.tied_encoders = self.algorithm.tied_encoders
        view1 = nn.Linear(cfg.data.kwargs.d1, cfg.latent_dim, bias=False)
        view2 = view1 if self.tied_encoders else nn.Linear(
            cfg.data.kwargs.d2, cfg.latent_dim, bias=False
        )
        self.encoders = [view1, view2]
        models = self.encoders[:1] if self.tied_encoders else self.encoders
        self.optimizers = [torch.optim.SGD(model.parameters(), **cfg.optimizer.kwargs) for model in models]
        self.algorithm.configure(self.encoders)
        self.encoders = [model.to(device=cfg.device, dtype=dtype) for model in self.encoders]
        self.algorithm.to(device=cfg.device, dtype=dtype)

    def step(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        state = self.algorithm.get_internal_state()
        output = encode(next(self.loader), self.encoders, state.stop_grad)
        loss = self.algorithm(output.x1, output.x2, state)
        loss.backward()
        for optimizer in self.optimizers:
            optimizer.step()
        return loss

    def run(self):
        set_seed(self.cfg.task.init_seed)
        with torch.no_grad():
            # Preserve the original initialization order, including both entries
            # when Self-distillation shares the same encoder between the views.
            for model in self.encoders:
                nn.init.normal_(model.weight, mean=0.0, std=self.cfg.task.init_scale)
        for step in range(self.cfg.task.total_steps):
            if step % 100 == 0:
                # Histories describe the parameters before this step's update.
                with torch.no_grad():
                    spectral = compute_spectral_metrics(
                        self.encoders, self.loader, self.algorithm, self.tied_encoders
                    )
            loss = self.step()
            if step % 100 == 0:
                metrics = {"loss": loss.item(), "iter": step}
                artifacts = {
                    "alignment": spectral.alignment.cpu().numpy(),
                    "spectrum_ratio": spectral.spectrum_ratio.cpu().numpy(),
                    "target_ratio": spectral.target_ratio.cpu().numpy(),
                    "iter": step,
                }
                self.logger.log_metrics(metrics, log_name="metrics")
                self.logger.log_artifacts(artifacts, f"alignments_{step}", artifact_type="pickle")
                pprint(metrics)
