import torch


def compute_covariance_svd(data_gen, encoder, view, num_batches=10):
    # 2. Estimate F = E[U(x)U(x)^T] using several fresh batches
    for iteration in range(num_batches):
        x_batch, y_batch = next(data_gen)
        if view == "x":
            data = x_batch
        elif view == "y":
            data = y_batch
        u = encoder(data)  # [batch, k]
        if iteration == 0:
            k = u.shape[1]
            Euu = torch.zeros((k, k), device=u.device)

        # Center u to compute the actual covariance if necessary
        Euu += torch.matmul(u.t(), u) / (u.shape[0] * num_batches)

    # 3. SVD of F to find its principal directions (eigenvectors)
    # Since F is a covariance matrix, it is symmetric; eigh is also an option.
    # U_f contains eigenvectors in columns.
    R_u, S_u, _ = torch.linalg.svd(Euu)

    out = {"R_u": R_u, "S_u": S_u, "Euu": Euu}
    return out


def compute_alignment(D, U):
    # 4. Compute Alignment: cosine_sim(u_i, D u_i)
    # We transform each eigenvector of F by D
    # U_f: [k, k] (columns are eigenvectors u_i)
    # D_ui: [k, k] (columns are the transformed vectors D @ u_i)
    D_ui = torch.matmul(D, U)

    # Calculate cosine similarity for each pair of columns
    # cos(theta) = (u_i . D_ui) / (||u_i|| * ||D_ui||)
    # Since u_i are eigenvectors of SVD, ||u_i|| = 1.
    dot_products = torch.sum(U * D_ui, dim=0)
    norms_D_ui = torch.norm(D_ui, dim=0)

    return dot_products / (norms_D_ui + 1e-8)


@torch.no_grad()
def compute_alignment_metrics(model, data_gen, num_batches=10):
    """
    Computes D = SQ and F = E[uu^T].
    Analyzes how well the eigenvectors of F align with the operator D.
    """
    # 1. Compute D = SQ
    # Note: Ensure Q and S are updated before calling this
    D = torch.matmul(model.S, model.Q)
    spectrum_x = compute_covariance_svd(data_gen, model.encoder_u, view="x", num_batches=num_batches)
    spectrum_y = compute_covariance_svd(data_gen, model.encoder_v, view="y", num_batches=num_batches)

    R_u = spectrum_x["R_u"]
    Evv = spectrum_y["Euu"]

    alignments_D = compute_alignment(D, R_u)
    alignments_S = compute_alignment(Evv, R_u)

    return {
        "alignments_D_F": alignments_D,       # List of k alignment scores
        "alignments_S_F": alignments_S,
        "eigenvalues_F": spectrum_x["S_u"],            # Importance of each direction
        "eigenvalues_S": spectrum_y["S_u"],            # Importance of each direction
    }


def compute_cse_spectrum(model, data_gen):
    svd_u = compute_covariance_svd(data_gen, model.encoder_u, view="x", num_batches=100)
    svd_v = compute_covariance_svd(data_gen, model.encoder_v, view="y", num_batches=100)

    _, svd, _ = torch.linalg.svd(svd_u["Euu"] + svd_v["Euu"])

    target_spectrum = torch.zeros_like(torch.zeros_like(svd, dtype=torch.float64))
    target_spectrum_tmp = model.compute_target_spectrum(data_gen)
    target_spectrum_tmp = torch.where(torch.sqrt(target_spectrum_tmp) < model.reg_c, 0.0, target_spectrum_tmp)

    n = target_spectrum_tmp.shape[0]
    k = svd.shape[0]
    num_to_copy = min(n, k)
    target_spectrum[:num_to_copy] = target_spectrum_tmp[:num_to_copy]

    spec = (svd + model.reg_c) ** 2
    spectrum = torch.where(torch.sqrt(target_spectrum) > model.reg_c, spec, svd)

    return spectrum, target_spectrum


def compute_semigrad_spectrum(model, data_gen):
    Sx = data_gen.info["S_1"]
    Sy = data_gen.info["S_2"]
    sigma = data_gen.info["sigma"]
    Corr = torch.abs(Sx * Sy) / (Sx ** 2 + sigma ** 2)
    Corr, _ = torch.sort(Corr, descending=True)

    thresh = model.reg_c / (Sx ** 2 + sigma ** 2)
    discriminant = Corr ** 2 - 4 * thresh
    ratio = torch.sqrt(((Sx ** 2 + sigma ** 2) / (Sy ** 2 + sigma ** 2)))

    svd = compute_covariance_svd(data_gen, model.encoder_u, view="x", num_batches=100)

    X = 1.0 / (Sx ** 2 + sigma ** 2)
    n = X.size(0)
    k = svd["S_u"].shape[0]
    num_to_copy = min(n, k)

    cov_X = torch.ones_like(svd["S_u"], dtype=svd["S_u"].dtype)
    cov_X[:num_to_copy] = X[:num_to_copy]

    singular_val = torch.sqrt((svd["S_u"] * cov_X))

    mask = (discriminant >= 0) & (thresh <= 0.25)

    mask = torch.nn.functional.pad(mask[:k], (0, max(0, k - n)), value=False)

    thresh = torch.nn.functional.pad(thresh[:k], (0, max(0, k - n)), mode="constant", value=0.0)
    ratio = torch.nn.functional.pad(ratio[:k], (0, max(0, k - n)), mode="constant", value=1.0)
    Corr = torch.nn.functional.pad(Corr[:k], (0, max(0, k - n)), mode="constant", value=0.0)
    target_spectrum = Corr / ratio
    spectrum = torch.where(
        mask,
        ratio * (singular_val + thresh / singular_val),
        svd["S_u"],
    )

    return spectrum, target_spectrum


def compute_unconstrained_cca_spectrum(model, data_gen):
    target_spectrum = model.compute_target_spectrum(data_gen)
    svd = compute_covariance_svd(data_gen, model.encoder_u, view="x", num_batches=100)
    spectrum = 2.0 * svd["S_u"]
    return spectrum, target_spectrum
