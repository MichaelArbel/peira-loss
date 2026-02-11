import torch
from linalg_utils import exact_inverse, add_ridge, solve_with_cholesky



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

def compute_covariance(encoder, data_gen, num_batches=10):
    # 2. Estimate F = E[U(x)U(x)^T] using several fresh batches
    for iteration in range(num_batches):
        data = next(data_gen)
        outputs = encode(data, encoders)  # [batch, k]
        if iteration == 0:
            k = outputs[0].shape[1]
            Covs = [torch.zeros((k, k), device=output.device) for i in len(outputs)+1]

        # Center u to compute the actual covariance if necessary
        Covs[0] +=  torch.matmul(outputs[1].t(), outputs[0]) / (outputs[0].shape[0] * num_batches)
        for i,output in enumerate(outputs):
            Covs[i+1] += torch.matmul(output.t(), output) / (output.shape[0] * num_batches)
  
    return Covs[0], Covs[1:]






def compute_cse_spectrum(covariance_spectrum, target_spectrum, reg):
    #_, cov_lists = compute_covariance(encoders, data_gen, num_batches=100)

    target_spectrum_padded = torch.zeros_like(torch.zeros_like(spectrum, dtype=torch.float64))
    target_spectrum_tmp = torch.where(torch.sqrt(target_spectrum) < reg, 0.0, target_spectrum)

    n = target_spectrum.shape[0]
    k = covariance_spectrum.shape[0]
    num_to_copy = min(n, k)
    target_spectrum_padded[:num_to_copy] = target_spectrum[:num_to_copy]

    spec = (covariance_spectrum + reg) ** 2
    spectrum = torch.where(torch.sqrt(target_spectrum_padded) > reg, spec, covariance_spectrum)

    return spectrum, target_spectrum_padded


def compute_semigrad_spectrum(covariance_spectrum, full_target_spectrum, reg):
    #Sx = data_gen.info["S_1"]
    #Sy = data_gen.info["S_2"]
    #sigma = data_gen.info["sigma"]

    # inter_spectrum =  torch.abs(Sx * Sy)
    # intra_spectrum_x = (Sx ** 2 + sigma ** 2)
    # intra_spectrum_y = (Sy ** 2 + sigma ** 2)

    inter_spectrum =  full_target_spectrum["inter_spectrum"]
    intra_spectrum_x = full_target_spectrum["intra_spectrum_1"]
    intra_spectrum_y = full_target_spectrum["intra_spectrum_2"]


    thresh = reg / intra_spectrum_x
    discriminant = Corr ** 2 - 4 * thresh
    ratio = torch.sqrt((intra_spectrum_x / intra_spectrum_y))

    X = 1.0 / intra_spectrum_x
    n = X.size(0)
    k = covariance_spectrum.shape[0]
    num_to_copy = min(n, k)

    cov_X = torch.ones_like(covariance_spectrum, dtype=covariance_spectrum.dtype)
    cov_X[:num_to_copy] = X[:num_to_copy]

    singular_val = torch.sqrt((covariance_spectrum * cov_X))

    mask = (discriminant >= 0) & (thresh <= 0.25)

    mask = torch.nn.functional.pad(mask[:k], (0, max(0, k - n)), value=False)

    thresh = torch.nn.functional.pad(thresh[:k], (0, max(0, k - n)), mode="constant", value=0.0)
    ratio = torch.nn.functional.pad(ratio[:k], (0, max(0, k - n)), mode="constant", value=1.0)
    Corr = torch.nn.functional.pad(Corr[:k], (0, max(0, k - n)), mode="constant", value=0.0)
    target_spectrum = Corr / ratio
    spectrum = torch.where(
        mask,
        ratio * (singular_val + thresh / singular_val),
        covariance_spectrum,
    )

    return spectrum, target_spectrum


def compute_unconstrained_cca_spectrum(covariance_spectrum, target_spectrum):  
    
    n = target_spectrum.shape[0]
    k = covariance_spectrum.shape[0]
    num_to_copy = min(n, k)
    target_spectrum_padded[:num_to_copy] = target_spectrum[:num_to_copy]

    return spectrum, target_spectrum_padded
