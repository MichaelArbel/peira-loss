import torch
from csrlearning_experiments.exp_types import SpectralMetrics
from csrlearning_experiments.training.pipeline import compute_covariance, encode
from csrlearning_experiments.evaluation.spectra import (
    compute_cse_spectrum,
    compute_semigrad_spectrum,
    compute_unconstrained_cca_spectrum,
)
from csrlearning import algorithms as algo_mod
from csrlearning.objectives import eckhart_young_loss, cse_loss, exact_linear_probe_loss, l2_penalty




def compute_alignment(M, U, B=None):
    # 4. Compute Alignment: cosine_sim(u_i, M u_i)
    # M: [k,k] matrix
    # U: [k,d] matrix 
    # B: [k,k] PSD matrix defining a metric  
    if B is not None:
        M_ui = M@B@U
        norms_M_ui = torch.sqrt(torch.diag(M_ui.t()@B@M_ui))
        dot_products = torch.sum((B@U)*M_ui, dim=0)
        norms_U = torch.sqrt(torch.diag(U.t()@B@U))
    else:
        M_ui = M@U
        norms_M_ui = torch.sqrt(torch.diag(M_ui.t()@M_ui))
        norms_U = torch.sqrt(torch.diag(U.t()@U))
        dot_products = torch.sum(U*M_ui, dim=0)

    # Calculate cosine similarity for each pair of columns
    # cos(theta) = (u_i . M_ui)_B / (||u_i||_B * ||M_ui||_B)
    

    return dot_products / ((norms_M_ui + 1e-8)*(norms_U + 1e-8))


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
    spectrum_ratio, target_ratio = compute_algorithm_spectrum(algorithm, S, target_spectrum_info)

    local_alignment = compute_alignment(inter_cov, R)
    params_1 = [param for param in encoders[0].parameters()]
    params_2 = [param for param in encoders[1].parameters()]

    if tied_encoders:

        W = params_1[0]
        B = target_spectrum_info.cov_1
        U = target_spectrum_info.singular_vectors_1
    else:
        W = torch.cat([params_1[0], params_2[0]], dim=1)
        B = torch.block_diag(target_spectrum_info.cov_1, target_spectrum_info.cov_2)
        U = torch.cat(
            [target_spectrum_info.singular_vectors_1, target_spectrum_info.singular_vectors_2],
            dim=0,
        )

    eigvals, eigvecs = torch.linalg.eigh(B)
    sqrt_B = eigvecs @ torch.diag(torch.sqrt(eigvals.clamp(min=0))) @ eigvecs.T

    M = sqrt_B @ W.t() @ W @ sqrt_B
    alignment = compute_alignment(M, U)

    upper_bound_reg  = compute_upper_bound_reg(algorithm, target_spectrum_info)

    #print(f"upper_bound_reg :{upper_bound_reg}")

    return SpectralMetrics(
        spectrum_ratio=spectrum_ratio,
        local_alignment=local_alignment,
        alignment=alignment,
        target_ratio=target_ratio,
    )


def compute_upper_bound_reg(algorithm, target_spectrum_info):
    if isinstance(algorithm, algo_mod.UnconstrainedCCAAlgorithm):
        return torch.inf
    if isinstance(algorithm, algo_mod.CSEAlgorithm):
        # |s_1s_2|/(sqrt(s_1** + sigma**2)sqrt(s_2** + sigma**2) )
        return torch.sqrt(target_spectrum_info.inter_spectrum/target_spectrum_info.intra_spectrum_1) 

    if isinstance(algorithm, algo_mod.SSLAlgorithm):
        # 1/4 |s_1s_2|**2/(s_1** + sigma**2)
        return 0.25*target_spectrum_info.inter_spectrum**2/target_spectrum_info.intra_spectrum_1




def compute_deterministic_loss(algorithm, X,Y, state):
    if isinstance(algorithm, algo_mod.UnconstrainedCCAAlgorithm):
        return eckhart_young_loss(X,Y)
    if isinstance(algorithm, algo_mod.CSEAlgorithm):
        return cse_loss(X,Y, None, algorithm.reg)
    if isinstance(algorithm, algo_mod.SSLAlgorithm):
        loss = exact_linear_probe_loss(X, Y, algorithm.reg, symmetric=False)
        loss = loss + 0.5 * algorithm.reg * l2_penalty(algorithm._weights)
        return loss

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
