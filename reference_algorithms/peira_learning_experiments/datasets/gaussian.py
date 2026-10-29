"""Fixed paired toy data with orthogonal latent and noise samples."""

import torch
from peira_learning_experiments.exp_types import DataInfo


class LinearCCADataGenerator:
    def __init__(self, d1=20, d2=20, k=10, sigma=0.1, batch_size=50,
                 device="cpu", dtype=torch.float64):
        self.A1, self.A2, self.info = generate_correlated_matrices(
            k, d1, d2, sigma, device, dtype
        )
        z, eps_1, eps_2 = generate_deterministic_data(
            b=batch_size, d=[k, d1, d2], device=device, dtype=dtype
        )
        self.data = (z @ self.A1.t() + sigma * eps_1,
                     z @ self.A2.t() + sigma * eps_2)

    def __iter__(self):
        return self

    def __next__(self):
        return self.data


def generate_correlated_matrices(dim_z, dim_x1, dim_x2, sigma, device, dtype):
    """Construct the fixed positive-correlation projections and their spectra."""
    S_vals_1 = torch.logspace(0, -4, steps=dim_z, device=device, dtype=dtype)
    S_vals_2 = 1.0 * S_vals_1
    k = len(S_vals_1)
    spectrum = torch.abs(S_vals_1 * S_vals_2) / (torch.sqrt(S_vals_1 ** 2 + sigma ** 2) * torch.sqrt(S_vals_2 ** 2 + sigma ** 2))
    spectrum, index = torch.sort(spectrum, descending=True)
    S_vals_1 = S_vals_1[index]
    S_vals_2 = S_vals_2[index]
    Q, _ = torch.linalg.qr(torch.randn(dim_z, k, device=device, dtype=dtype))
    P1, _ = torch.linalg.qr(torch.randn(dim_x1, k, device=device, dtype=dtype))
    A1 = P1 @ torch.diag(S_vals_1) @ Q.t()
    A2 = P1 @ torch.diag(S_vals_2) @ Q.t()
    cov_1 = P1 @ torch.diag(S_vals_1 ** 2 + sigma ** 2) @ P1.t()
    cov_2 = P1 @ torch.diag(S_vals_2 ** 2 + sigma ** 2) @ P1.t()
    info = DataInfo(singular_vectors_1=P1, singular_vectors_2=P1, cov_1=cov_1, cov_2=cov_2, intra_spectrum_1=S_vals_1 ** 2 + sigma ** 2, inter_spectrum=torch.abs(S_vals_2 * S_vals_1), spectrum=spectrum)
    return (A1, A2, info)

def generate_deterministic_data(b=512, d=[10, 128, 128], device='cpu', dtype=torch.float64):
    d_tot = sum(d)
    raw_z = torch.randn(b, d_tot, dtype=torch.float64, device=device)
    Q, _ = torch.linalg.qr(raw_z, mode='reduced')
    data = Q * torch.sqrt(torch.tensor(b, dtype=torch.float64))
    z = data[:, :d[0]]
    eps_1 = data[:, d[0]:d[1] + d[0]]
    eps_2 = data[:, d[1] + d[0]:]
    return (z, eps_1, eps_2)
