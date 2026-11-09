"""PEIRA: anti-collapse loss with closed-form ridge predictor and running covariances."""

from peira_loss.loss import JointSelfDistillationLoss, PEIRALoss, SelfDistillationLoss

__all__ = ["PEIRALoss", "SelfDistillationLoss", "JointSelfDistillationLoss"]
__version__ = "0.2.0"
