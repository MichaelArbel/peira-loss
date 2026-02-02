import os
import importlib


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