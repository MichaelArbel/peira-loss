import math
import torch

def set_seed(seed):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)




def compute_running_average(avg_field_old, new_field, eta):
    if avg_field_old is None:
        return new_field
    return (1 - eta) * avg_field_old + eta * new_field


def cosine_momentum_schedule(epoch, t_max, m_start, m_end):
    return m_end + 0.5 * (m_start - m_end) * (1 + math.cos(math.pi * epoch / t_max))
