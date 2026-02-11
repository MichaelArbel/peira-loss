import torch

from .core_types import EncodedBatch


def encode(data, encoders, stop_grad=False):
    first_output = [encoders[0](data[0])]
    if stop_grad:
        with torch.no_grad():
            last_output = [encoders[i](data[i]) for i in range(1, len(data))]
    else:
        last_output = [encoders[i](data[i]) for i in range(1, len(data))]
    output = tuple(first_output + last_output)
    return EncodedBatch(output)


def compute_covariance(encode_fn, data_gen, num_batches=10):
    # 2. Estimate F = E[U(x)U(x)^T] using several fresh batches
    for iteration in range(num_batches):
        data = next(data_gen)
        outputs = encode_fn(data)
        views = outputs.views
        num_cov = len(views) + 1
        if iteration == 0:
            k = views[0].shape[1]
            Covs = [torch.zeros((k, k), device=views[0].device) for _ in range(num_cov)]

        # Center u to compute the actual covariance if necessary
        Covs[0] += torch.matmul(views[1].t(), views[0]) / (views[0].shape[0] * num_batches)
        for i, output in enumerate(views):
            Covs[i + 1] += torch.matmul(output.t(), output) / (output.shape[0] * num_batches)

    return Covs[0], Covs[1:]

