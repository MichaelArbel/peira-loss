import torch
import math


def cosine_momentum_schedule(epoch, t_max, m_start, m_end):
    return m_end + 0.5 * (m_start - m_end) * (1 + math.cos(math.pi * epoch / t_max))


def compute_regression_loss(u, v):
    # Placeholder for L_reg(U, V)
    return 0.5 * torch.mean(torch.sum((u - v) ** 2, dim=1))


def compute_auxiliary_loss(u, v, Q, S, reg_c):
    # Placeholder for L_aux(U, V, Q, S)
    # Implements a form of the Contrastive Spectral Objective
    # tr(U^T Q S Q U) or similar logic derived from your CSE research
    Qu = torch.einsum("kl,ml->mk", Q, u)

    loss = (1.0 / 2) * torch.einsum("mk,mk->", (Qu @ S), Qu) / u.shape[0]
    loss += 0.5 * reg_c * torch.einsum("mk,mk->", u, u) / u.shape[0]
    loss += -torch.einsum("mk,mk->", Qu, v) / u.shape[0]

    return loss


def compute_cse_objective(Q, S, u, reg_c):
    return (1.0 / 2) * torch.sum(S * Q) - 0.5 * reg_c * torch.einsum("mk,mk->", u, u) / u.shape[0]


def compute_semi_gradient_ssl_loss(u, v, P, reg_c, model):
    l2_penalty = 0
    for param in model.parameters():
        l2_penalty += torch.sum(param ** 2)
    pred = torch.einsum("rl,bl->br", P, u)
    return 0.5 * torch.mean(torch.sum((pred - v) ** 2, dim=1)) + 0.5 * reg_c * l2_penalty


def compute_running_average_field(avg_field_old, new_field, eta):
    if avg_field_old is None:
        return new_field
    return (1 - eta) * avg_field_old + eta * new_field
