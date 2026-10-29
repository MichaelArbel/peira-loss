"""Spectral ratios and ground-truth eigenvector alignment."""

import torch
from peira_learning import algorithms as algo_mod
from peira_learning_experiments.exp_types import SpectralMetrics
from peira_learning_experiments.training.pipeline import compute_covariance, encode
from peira_learning_experiments.evaluation.spectra import compute_peira_spectrum, compute_self_distillation_spectrum


def compute_alignment(M, U):
    """Cosine alignment of each ground-truth direction with its image under M."""
    M_ui = M @ U
    norms_M_ui = torch.sqrt(torch.diag(M_ui.t() @ M_ui))
    norms_U = torch.sqrt(torch.diag(U.t() @ U))
    dot_products = torch.sum(U * M_ui, dim=0)
    return dot_products / ((norms_M_ui + 1e-8) * (norms_U + 1e-8))


def compute_spectral_metrics(encoders, data_source, algorithm, tied_encoders):
    cov_lists = compute_covariance(lambda data: encode(data, encoders, False), data_source)
    intra_cov = cov_lists[0]
    if not tied_encoders:
        intra_cov = intra_cov + cov_lists[1]
    _, S, _ = torch.linalg.svd(intra_cov)
    target_spectrum_info = data_source.info
    spectrum_ratio, target_ratio = compute_algorithm_spectrum(algorithm, S, target_spectrum_info)
    params_1 = [param for param in encoders[0].parameters()]
    params_2 = [param for param in encoders[1].parameters()]
    if tied_encoders:
        W = params_1[0]
        B = target_spectrum_info.cov_1
        U = target_spectrum_info.singular_vectors_1
    else:
        W = torch.cat([params_1[0], params_2[0]], dim=1)
        B = torch.block_diag(target_spectrum_info.cov_1, target_spectrum_info.cov_2)
        U = torch.cat([target_spectrum_info.singular_vectors_1, target_spectrum_info.singular_vectors_2], dim=0)
    eigvals, eigvecs = torch.linalg.eigh(B)
    sqrt_B = eigvecs @ torch.diag(torch.sqrt(eigvals.clamp(min=0))) @ eigvecs.T
    M = sqrt_B @ W.t() @ W @ sqrt_B
    alignment = compute_alignment(M, U)
    return SpectralMetrics(spectrum_ratio=spectrum_ratio, alignment=alignment, target_ratio=target_ratio)

def compute_algorithm_spectrum(algorithm, cov_spectrum, target_spectrum_info):
    if isinstance(algorithm, algo_mod.PEIRAAlgorithm):
        target_spectrum = target_spectrum_info.spectrum
        return compute_peira_spectrum(cov_spectrum, target_spectrum, algorithm.reg)
    if isinstance(algorithm, algo_mod.SelfDistillationAlgorithm):
        return compute_self_distillation_spectrum(cov_spectrum, target_spectrum_info, algorithm.reg)
    raise ValueError(f'Unsupported algorithm type for spectrum: {type(algorithm).__name__}')
