"""Resolve configured dataset and algorithm classes."""

import importlib


def instantiate(class_path):
    module, name = class_path.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)
