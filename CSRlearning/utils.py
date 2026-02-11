import os
import importlib
import math
import torch

from csr_types import EncodedBatch

def set_seed(seed):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def instantiate(module_name):
    module, attr = os.path.splitext(module_name)
    try:
        module = importlib.import_module(module)
        return getattr(module, attr[1:])
    except BaseException:
        try:
            module = instantiate(module)
            return getattr(module, attr[1:])
        except BaseException:
            return eval(module + attr[1:])


def compute_running_average(avg_field_old, new_field, eta):
    if avg_field_old is None:
        return new_field
    return (1 - eta) * avg_field_old + eta * new_field


def cosine_momentum_schedule(epoch, t_max, m_start, m_end):
    return m_end + 0.5 * (m_start - m_end) * (1 + math.cos(math.pi * epoch / t_max))

def encode(data, encoders, stop_grad = False):
    first_output = [encoders[0](data[0])]
    if stop_grad:
        with torch.no_grad():
            last_output = [encoders[i](data[i]) for i in range(1,len(data))]
    else:
        last_output = [encoders[i](data[i]) for i in range(1,len(data))]
    output = tuple(first_output + last_output)
    return EncodedBatch(output)

class MomentumOptimizer:
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr
    def step(self):
        for param in self.params:
            param.data -=  self.lr*param.grad
    def zero_grad(self):
        for param in self.params:
            param.grad = None

class CosineScheduler:
    def __init__(self, optimizer, T_max, m_end=0.0):
        self.counter = 0
        self.T_max = T_max
        self.m_start = optimizer.lr
        self.m_end = m_end
        self.m_current = self.m_start
        self.optimizer = optimizer
    def step(self):
        self.counter +=1
        self.m_current = cosine_momentum_schedule(self.counter, 
                                                self.T_max, 
                                                self.m_start, 
                                                self.m_end
                                                )
        self.optimizer.lr =  self.m_current
