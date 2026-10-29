# Reference algorithms

This directory contains the original PEIRA implementation, PEIRA loss variants,
baseline methods, and alternative optimization algorithms. It also includes
reproducible deterministic toy experiments and figure-generation scripts.
These implementations remain usable for studying and comparing the algorithms.

For integration into other projects, use the recommended `peira_loss` package in
[`src/peira_loss`](../src/peira_loss), following the installation and usage
instructions in the [main README](../README.md).

The reference library is provided as `peira_learning`; the toy experiments and
plotting utilities are in
[`peira_learning_experiments`](peira_learning_experiments/README.md).

## `PEIRAStoCompGrad` for Representation Learning

`PEIRAStoCompGrad` is a stateful loss module for learning representations from two aligned latent views. Typical examples are:

- two augmentations of the same image,
- two matched modalities,
- a predicted latent state and its target latent state in a sequential or planning model.

The class expects two tensors `X` and `Y` of shape `[batch_size, latent_dim]`. It keeps an internal linear predictor that is updated online from the current mini-batches, and it returns a differentiable loss that you use to train the encoder that produced `X` and `Y`.

If your model produces sequences with shape `[batch_size, time, latent_dim]`, the usual pattern is to apply the loss to aligned pairs one time step at a time, or to reshape the sequence into a larger batch of aligned pairs before calling the module.

Import it with:

```python
from peira_learning.algorithms import PEIRAStoCompGrad
```

## Minimal Usage

```python
import torch
from peira_learning.algorithms import PEIRAStoCompGrad

device = "cuda"
latent_dim = 256
total_updates = 100_000

encoder = ...
optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-4)

loss_fn = PEIRAStoCompGrad(
    latent_dim=latent_dim,
    reg=0.3,
    n_iter=1,
    lr=0.9,
    T_max=total_updates,
    lr_min=0.5,
    reg_min=None,
).to(device)

for batch in loader:
    view1, view2 = batch

    z1 = encoder(view1.to(device))
    z2 = encoder(view2.to(device))

    loss_dict = loss_fn(z1, z2)
    loss = loss_dict["loss"]

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if loss_fn.reg_scheduler is not None:
        loss_fn.reg_scheduler.step()

    monitor_value = loss_dict["peira_loss"].detach()
```

## What the Forward Pass Returns

`loss_fn(X, Y)` returns a dictionary with two entries:

- `"loss"`: the training loss to backpropagate through the encoder.
- `"peira_loss"`: the PEIRA objective evaluated with the current internal predictor. This is useful for logging and monitoring, but in the usual setup you optimize `"loss"`.

In other words, training normally looks like:

```python
loss_dict = loss_fn(X, Y)
loss_dict["loss"].backward()
```

## Important Implementation Details

- `PEIRAStoCompGrad` is a loss module, not part of your encoder. Your main optimizer should optimize the encoder or projector parameters, not the internal regressor stored inside `loss_fn`.
- The internal regressor is updated inside `PEIRAStoCompGrad.forward(...)` with its own SGD optimizer.
- Those internal updates are done on detached `float32` copies of `X` and `Y`, so mixed-precision encoder training is fine.
- `PEIRAStoCompGrad` always uses the symmetric PEIRA setup and always maintains both the predictor and its adjoint internally.
- The module is stateful, so you should create it once and reuse it throughout training. Re-creating it every iteration would reset its internal covariance estimates and optimizer state.

## Hyperparameters

### `latent_dim`

Dimension of the latent representation. If your encoder outputs tensors of shape `[B, D]`, then `latent_dim` must be `D`.

Internally, the class maintains square matrices of shape `[latent_dim, latent_dim]`, so this value controls both compatibility and memory/computation cost.

### `reg`

The regularization strength used in the PEIRA objective and in the ridge term added before inverting the covariance estimate.

Practical effect:

- larger values make the inverse better conditioned and usually stabilize training,
- smaller values reduce bias but can make the predictor update noisier or less stable.

### `n_iter`

Number of internal regressor updates performed every time you call `loss_fn(X, Y)`.

Practical effect:

- `n_iter=1` is the standard choice and is the default,
- larger values make the internal predictor adapt more aggressively to the current batch,
- larger values also increase compute because the internal update is repeated `n_iter` times per call.

Because the same pair `(X, Y)` is reused inside one forward call, increasing `n_iter` means taking several internal updates from the same mini-batch.

### `lr`

Learning rate of the internal SGD optimizer used to update the covariance tracker / internal linear regressor.

Practical effect:

- larger values make the internal state react faster to new batches,
- smaller values make it smoother but slower to adapt.

This learning rate is separate from the optimizer you use for the encoder.

### `T_max`

Horizon of the cosine annealing schedule used for the internal optimizer learning rate. If `reg_min` is set, the same horizon is also used for the regularization schedule.

The most useful rule is:

- set `T_max` to the total number of times you expect the module to update its internal state.

If you call `loss_fn` once per training step, `T_max` is usually the total number of training steps. If you apply the same loss once per step of an unrolled sequence, `T_max` should count all of those per-step calls.

### `lr_min`

Final learning rate reached by the cosine annealing schedule of the internal optimizer.

Practical effect:

- if `lr_min` is close to `lr`, the internal optimizer stays aggressive throughout training,
- if `lr_min` is much smaller, the internal predictor update becomes progressively more conservative.

### `reg_min`

Optional final value for a cosine annealing schedule on `reg`.

Practical effect:

- if `reg_min=None`, the regularization strength stays fixed at `reg`,
- if `reg_min` is a float, `reg` is annealed from its initial value down to `reg_min`.

When you use `reg_min`, call:

```python
if loss_fn.reg_scheduler is not None:
    loss_fn.reg_scheduler.step()
```

once per outer training update.

## Choosing `T_max` in Common Setups

For two-view image representation learning:

- one call to `loss_fn` per optimizer step usually means `T_max = number_of_training_steps`.

For sequential or planning-style representation learning:

- if you reuse the same `PEIRAStoCompGrad` instance across multiple latent pairs in each rollout, count every call,
- a common choice is `T_max = number_of_training_steps * number_of_loss_applications_per_step`.

## Summary of the Public API

```python
PEIRAStoCompGrad(
    latent_dim,
    reg,
    n_iter=1,
    lr=0.01,
    T_max=20000,
    lr_min=0.0,
    reg_min=None,
)
```

The arguments above are the ones that matter for the behavior of `PEIRAStoCompGrad`. Extra keyword arguments are accepted by the constructor for API compatibility, but they are not used by this class directly.
