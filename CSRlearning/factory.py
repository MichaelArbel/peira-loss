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


def build_optimizer_bundle(encoders, tied_encoders, optimizer_spec, scheduler_spec):
    from core_types import OptimizerBundle

    first_encoder = encoders[0]
    if tied_encoders:
        for i in range(len(encoders)):
            encoders[i] = first_encoder

    optimizer_ctor = lambda params: instantiate(optimizer_spec.name)(params, **optimizer_spec.kwargs)
    scheduler_ctor = lambda optimizer: instantiate(scheduler_spec.name)(
        optimizer, **scheduler_spec.kwargs)

    if not tied_encoders:
        optimizers = [optimizer_ctor(encoder.parameters()) for encoder in encoders]
    else:
        optimizers = [optimizer_ctor(first_encoder.parameters())]

    schedulers = [scheduler_ctor(opt) for opt in optimizers]

    bundle = OptimizerBundle()
    for opt, sch in zip(optimizers, schedulers):
        bundle.add(opt, sch, managed_by_algorithm=False)

    return encoders, bundle
