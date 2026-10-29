import torch

from peira_learning_experiments.exp_types import EncodedBatch


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
    """Preserve the original ten-batch accumulation for spectrum normalization."""
    for iteration in range(num_batches):
        views = encode_fn(next(data_gen)).views
        if iteration == 0:
            k = views[0].shape[1]
            covariances = [torch.zeros((k, k), device=views[0].device) for _ in views]
        for covariance, output in zip(covariances, views):
            covariance += output.t() @ output / (output.shape[0] * num_batches)
    return covariances
