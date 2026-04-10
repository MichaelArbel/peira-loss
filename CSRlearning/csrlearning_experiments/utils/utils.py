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

import torch
from torch.optim.lr_scheduler import _LRScheduler

class InverseTimeLRScheduler(_LRScheduler):
    """
    lr(t) = lr0 / (t + 1), where t increments on each scheduler.step().
    - t=0 => lr = lr0
    - t=1 => lr = lr0/2
    - ...
    """
    def __init__(self, optimizer, lr0=None, last_epoch=-1, power=1., verbose=False):
        # If lr0 not provided, use optimizer's initial lr(s) as lr0.
        self._user_lr0 = lr0
        self.power = power
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        # self.last_epoch is the internal step counter:
        # - starts at -1
        # - after first step() becomes 0  -> t=0
        t = self.last_epoch
        factor = 1.0 / (0.000001*t + 1.0)**self.power

        if self._user_lr0 is None:
            # Use each param group's base_lr as lr0
            return [base_lr * factor for base_lr in self.base_lrs]
        else:
            # Use a single lr0 for all param groups
            return [float(self._user_lr0) * factor for _ in self.base_lrs]

