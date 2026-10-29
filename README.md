# peira-loss

Pure-PyTorch implementation of the PEIRA loss (Arbel, Terver, Ponce, NeurIPS 2026). PEIRA
prevents representation collapse in joint-embedding self-supervised learning with the
closed-form ridge-regression predictor between the two views, `P = Sigma (N + lambda I)^-1`,
built from exponential moving averages of the views' second moments (EMA rate cosine-annealed
from `lr` to `lr_min`), plus a ridge `lambda` on the representations. The moving averages and
the step counter live in the module `state_dict`, so they are saved and restored with a
checkpoint. Multi-GPU training works out of the box: batch second moments are summed across
ranks with `all_reduce`.

```bash
pip install -e .
```

```python
import torch
from peira_loss import PEIRALoss

loss_fn = PEIRALoss(latent_dim=256, T_max=total_steps, reg=0.1, lr=0.9, lr_min=0.5)  # ImageNet-1K recipe
z1, z2 = projector(encoder(view1)), projector(encoder(view2))  # [B, 256] each
out = loss_fn(z1, z2)          # dict: loss, peira_loss, trace_P, cse_loss, reg_loss, peira_lr
out["loss"].backward()
```

`predictor_cov` selects the `Sigma` used in `P`: `"batch"` (default) uses the current
batch's estimate, as in the runs reported in the paper; `"ema"` uses the moving average, as
written in Algorithm 1. `N` is always the moving average.

Call the loss exactly once per training step: each `forward` advances the EMA-rate
schedule.

## Citation

```bibtex
@inproceedings{arbel2026peira,
  title     = {{PEIRA: Learning Predictive Encoders through Inter-View Regressor Alignment}},
  author    = {Arbel, Michael and Terver, Basile and Ponce, Jean},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
  eprint    = {2605.17671},
  archivePrefix = {arXiv}
}
```
