from types import SimpleNamespace

import torch

from core.algorithms import UnconstrainedCCAAlgorithm
from factory import build_optimizer_bundle
from generator import LinearCCADataGenerator
from core.pipeline import encode


def main():
    device = "cpu"
    dtype = torch.float32

    data_source = LinearCCADataGenerator(
        d1=20,
        d2=20,
        k=10,
        sigma=0.1,
        batch_size=256,
        device=device,
        dtype=dtype,
        deterministic=False,
    )

    latent_dim = 10
    encoder_u = torch.nn.Linear(20, latent_dim, bias=False).to(device=device, dtype=dtype)
    encoder_v = torch.nn.Linear(20, latent_dim, bias=False).to(device=device, dtype=dtype)
    encoders = [encoder_u, encoder_v]

    optimizer_spec = SimpleNamespace(
        name="torch.optim.SGD",
        kwargs={"lr": 1e-2},
    )
    scheduler_spec = SimpleNamespace(
        name="torch.optim.lr_scheduler.CosineAnnealingLR",
        kwargs={"T_max": 1000},
    )

    encoders, optimizer_bundle = build_optimizer_bundle(
        encoders=encoders,
        tied_encoders=False,
        optimizer_spec=optimizer_spec,
        scheduler_spec=scheduler_spec,
    )

    algorithm = UnconstrainedCCAAlgorithm()
    algorithm.configure(encoders, optimizer_bundle)

    steps = 1000
    for step in range(steps):
        state = algorithm.update_internal_state(data_source)
        optimizer_bundle.zero_grad()

        batch = next(data_source)
        outputs = encode(batch, encoders, state.stop_grad)
        loss = algorithm.compute_loss(outputs.x1, outputs.x2, state)
        loss.backward()
        optimizer_bundle.step()

        if step % 100 == 0:
            print(f"step={step} loss={loss.item():.6f}")


if __name__ == "__main__":
    main()
