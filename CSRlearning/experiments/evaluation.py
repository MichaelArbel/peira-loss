import torch
from experiments.exp_types import SpectralMetrics
from experiments.pipeline import compute_covariance, encode
from csrlearning.spectra import (
    compute_cse_spectrum,
    compute_semigrad_spectrum,
    compute_unconstrained_cca_spectrum,
)
from csrlearning import algorithms as algo_mod

def compute_alignment(M, U, B=None):
    # 4. Compute Alignment: cosine_sim(u_i, M u_i)
    # M: [k,k] matrix
    # U: [k,d] matrix that are B-orthonormal
    # B: [k,k] PSD matrix defining a metric  
    if B is not None:
        M_ui = M@B@U
        norms_M_ui = torch.sqrt(torch.diag(M_ui.t()@B@M_ui))
        dot_products = torch.sum((B@U)*M_ui, dim=0)
    else:
        M_ui = M@U
        norms_M_ui = torch.sqrt(torch.diag(M_ui.t()@M_ui))
        dot_products = torch.sum(U*M_ui, dim=0)

    # Calculate cosine similarity for each pair of columns
    # cos(theta) = (u_i . M_ui)_B / (||u_i||_B * ||M_ui||_B)
    # Since u_i are B-orthonormal, ||u_i||_B = 1.
    

    return dot_products / (norms_M_ui + 1e-8)


def compute_spectral_metrics(encoders, data_source, algorithm, tied_encoders):
    if data_source.info is None:
        return None
    inter_cov, cov_lists = compute_covariance(
        lambda data: encode(data, encoders, False),
        data_source,
    )

    intra_cov = cov_lists[0]
    if not tied_encoders:
        intra_cov = intra_cov + cov_lists[1]
        inter_cov = inter_cov + inter_cov.t()

    R, S, _ = torch.linalg.svd(intra_cov)
    target_spectrum_info = data_source.info
    spectrum, target_spectrum = compute_algorithm_spectrum(algorithm, S, target_spectrum_info)

    local_alignment = compute_alignment(inter_cov, R)

    if tied_encoders:
        W = encoders[0].weight
        B = target_spectrum_info.cov_1
        U = target_spectrum_info.singular_vectors_1
    else:
        W = torch.cat([encoders[0].weight, encoders[1].weight], dim=0)
        B = torch.block_diag(target_spectrum_info.cov_1, target_spectrum_info.cov_2)
        U = torch.cat(
            [target_spectrum_info.singular_vectors_1, target_spectrum_info.singular_vectors_2],
            dim=0,
        )

    eigvals, eigvecs = torch.linalg.eigh(B)
    sqrt_B = eigvecs @ torch.diag(torch.sqrt(eigvals.clamp(min=0))) @ eigvecs.T

    M = sqrt_B @ W.t() @ W @ sqrt_B
    alignment = compute_alignment(M, U)

    return SpectralMetrics(
        spectrum=spectrum,
        target_spectrum=target_spectrum,
        local_alignment=local_alignment,
        alignment=alignment,
    )


def compute_algorithm_spectrum(algorithm, cov_spectrum, target_spectrum_info):
    if isinstance(algorithm, algo_mod.UnconstrainedCCAAlgorithm):
        target_spectrum = target_spectrum_info.spectrum
        return compute_unconstrained_cca_spectrum(cov_spectrum, target_spectrum)

    if isinstance(algorithm, algo_mod.CSEAlgorithm):
        target_spectrum = target_spectrum_info.spectrum
        return compute_cse_spectrum(cov_spectrum, target_spectrum, algorithm.reg)

    if isinstance(algorithm, algo_mod.SSLAlgorithm):
        return compute_semigrad_spectrum(cov_spectrum, target_spectrum_info, algorithm.reg)

    raise ValueError(f"Unsupported algorithm type for spectrum: {type(algorithm).__name__}")
