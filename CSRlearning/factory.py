import os
import importlib


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
