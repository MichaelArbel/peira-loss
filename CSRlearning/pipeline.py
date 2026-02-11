import torch

from core_types import EncodedBatch, SpectralMetrics
from evaluation import compute_alignment


def encode(data, encoders, stop_grad=False):
    first_output = [encoders[0](data[0])]
    if stop_grad:
        with torch.no_grad():
            last_output = [encoders[i](data[i]) for i in range(1, len(data))]
    else:
        last_output = [encoders[i](data[i]) for i in range(1, len(data))]
    output = tuple(first_output + last_output)
    return EncodedBatch(output)


def compute_covariance(encode_fn, data_gen, num_batches=10):
    # 2. Estimate F = E[U(x)U(x)^T] using several fresh batches
    for iteration in range(num_batches):
        data = next(data_gen)
        outputs = encode_fn(data)
        views = outputs.views
        num_cov = len(views) + 1
        if iteration == 0:
            k = views[0].shape[1]
            Covs = [torch.zeros((k, k), device=views[0].device) for _ in range(num_cov)]

        # Center u to compute the actual covariance if necessary
        Covs[0] += torch.matmul(views[1].t(), views[0]) / (views[0].shape[0] * num_batches)
        for i, output in enumerate(views):
            Covs[i + 1] += torch.matmul(output.t(), output) / (output.shape[0] * num_batches)

    return Covs[0], Covs[1:]


def compute_spectral_metrics(encoders, data_source, algorithm, tied_encoders):
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
    spectrum, target_spectrum = algorithm.compute_spectrum(S, target_spectrum_info)

    local_alignment = compute_alignment(inter_cov, R)

    if tied_encoders:
        W = encoders[0].weight
        B = target_spectrum_info.cov_1
        U = target_spectrum_info.singular_vectors_1
    else:
        W = torch.cat([encoders[0].weight, encoders[1].weight], dim=0)
        B = torch.cat([target_spectrum_info.cov_1, target_spectrum_info.cov_2], dim=0)
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
