import torch


def compute_cse_spectrum(covariance_spectrum, target_spectrum, reg):
    n = target_spectrum.shape[0]
    k = covariance_spectrum.shape[0]
    num_to_copy = min(n, k)
    target_spectrum_padded = torch.zeros_like(
        torch.zeros_like(covariance_spectrum, dtype=torch.float64)
    )
    target_spectrum_padded[:num_to_copy] = target_spectrum[:num_to_copy]


    target_spectrum_padded= torch.sqrt(target_spectrum_padded)-reg

    mask = target_spectrum_padded>0
    
    spectrum_ratio = torch.where(
            mask,
            covariance_spectrum/target_spectrum_padded,
            covariance_spectrum
        )
    #print(lol)
    return spectrum_ratio


# def compute_semigrad_spectrum(covariance_spectrum, full_target_spectrum, reg):
#     inter_spectrum = full_target_spectrum.inter_spectrum
#     intra_spectrum_x = full_target_spectrum.intra_spectrum_1
#     intra_spectrum_y = full_target_spectrum.intra_spectrum_2

#     Corr = inter_spectrum / intra_spectrum_x
#     thresh = reg / intra_spectrum_x
#     discriminant = Corr ** 2 - 4 * thresh
#     ratio = torch.sqrt((intra_spectrum_x / intra_spectrum_y))

#     X = 1.0 / intra_spectrum_x
#     n = X.size(0)
#     k = covariance_spectrum.shape[0]
#     num_to_copy = min(n, k)

#     cov_X = torch.ones_like(covariance_spectrum, dtype=covariance_spectrum.dtype)
#     cov_X[:num_to_copy] = X[:num_to_copy]

#     singular_val = torch.sqrt((covariance_spectrum * cov_X))

#     mask = (discriminant >= 0) & (thresh <= 0.25)

#     mask = torch.nn.functional.pad(mask[:k], (0, max(0, k - n)), value=False)

#     thresh = torch.nn.functional.pad(thresh[:k], (0, max(0, k - n)), mode="constant", value=0.0)
#     ratio = torch.nn.functional.pad(ratio[:k], (0, max(0, k - n)), mode="constant", value=1.0)
#     Corr = torch.nn.functional.pad(Corr[:k], (0, max(0, k - n)), mode="constant", value=0.0)
#     target_spectrum = Corr / ratio
#     spectrum = torch.where(
#         mask,
#         ratio * (singular_val + thresh / singular_val),
#         covariance_spectrum,
#     )

#     return spectrum, target_spectrum






def compute_semigrad_spectrum(covariance_spectrum, full_target_spectrum, reg):
    inter_spectrum = full_target_spectrum.inter_spectrum
    intra_spectrum_x = full_target_spectrum.intra_spectrum_1
    intra_spectrum_y = full_target_spectrum.intra_spectrum_2

    Corr = inter_spectrum / intra_spectrum_x
    thresh = reg / intra_spectrum_x
    discriminant = Corr ** 2 - 4 * thresh
    n = intra_spectrum_x.size(0)
    k = covariance_spectrum.shape[0]
    num_to_copy = min(n, k)
    mask = (discriminant >= 0) & (thresh <= 0.25)
    val = torch.where(
            mask,
            0.5 * (Corr + torch.sqrt(torch.clamp(discriminant, min=0))),
            torch.zeros_like(Corr, dtype=torch.float64)
        )

    target = val**2*intra_spectrum_x

    target_spectrum = torch.nn.functional.pad(target[:k], (0, max(0, k - n)), mode="constant", value=0.0)
    mask = torch.nn.functional.pad(mask[:k], (0, max(0, k - n)), value=False)
    spectrum_ratio = torch.where(
            mask,
            covariance_spectrum/target_spectrum,
            covariance_spectrum
        )
    #print(lol)
    return spectrum_ratio#, torch.ones_like(spectrum_ratio)









def compute_unconstrained_cca_spectrum(covariance_spectrum, target_spectrum):
    n = target_spectrum.shape[0]
    k = covariance_spectrum.shape[0]
    num_to_copy = min(n, k)
    target_spectrum_padded = torch.zeros_like(covariance_spectrum, dtype=torch.float64)
    target_spectrum_padded[:num_to_copy] = target_spectrum[:num_to_copy]
    spectrum_ratio = torch.where(
            target_spectrum_padded>0.,
            covariance_spectrum/target_spectrum_padded,
            covariance_spectrum
        )


    return spectrum_ratio
