# Deterministic toy experiments

This directory contains the experiment and plotting code for four figures:
`spectrum_image_peira.png`, `spectrum_image_self_distillation.png`,
`alignment_image_peira.png`, and `alignment_image_self_distillation.png`.
The algorithms themselves are provided by the core `peira_learning` package.

## Installation

From the `reference_algorithms` directory, activate your Python environment and run:

```bash
python -m pip install -e .
python -m pip install -e ./peira_learning_experiments
```

MLXP launches the configuration sweeps, logs each run, manages code snapshots,
and reads the saved artifacts for plotting. The experiment package declares MLXP,
NumPy, and Matplotlib; installing only the core package does not install these
experiment dependencies.
Use the same environment for training and plotting. No terminal image viewer is
required; figures are saved directly as PNG files.

## Launch the runs

From the repository root:

```bash
bash peira_learning_experiments/scripts/toy_deterministic.sh
```

The script changes to `peira_learning_experiments` before invoking `main.py`.
It enables MLXP version management and interactive mode; respond to any prompts.
The sweep launches ten runs:

| Method | Algorithm override | Regularization | Learning rate | Initialization scales |
| --- | --- | --- | --- | --- |
| Self-distillation | `self_distillation_exact` | `1e-5` | `10` | `1e-6`, `1e-5`, `1e-4`, `1e-3`, `1e-2` |
| PEIRA | `peira_exact` | `0.4` | `0.1` | `1e-6`, `1e-5`, `1e-4`, `1e-3`, `1e-2` |

All runs use seed `0`, initialization seed `0`, `400000` updates, CPU double
precision, linear encoders with 30 output dimensions, and constant learning rates.
The two views have dimension 20 and share 10 latent components, with noise level
`0.1` and a fixed batch of 50 orthogonalized latent/noise samples. PEIRA uses separate
encoders; Self-distillation shares its encoder across views and stops the gradient through the
second view. Initialization uses Gaussian weights with the specified scale.

For Slurm, adapt the account, resource limits, and Python environment, then submit:

```bash
cd peira_learning_experiments
sbatch scripts/toy_deterministic.sh
```

The launcher stores runs in `../../data/outputs/deterministic_2` and MLXP work
directories in `../../data/.workdir`, relative to `peira_learning_experiments`.
For this checkout layout these are `CSRLeaning/data/outputs/deterministic_2` and
`CSRLeaning/data/.workdir`. Edit `parent_log_dir` in the launcher to use a fresh
run directory; use that same path with the plotting command's `--runs-dir` option.

Each run logs loss and iteration metrics, plus `alignments_<step>` pickle artifacts
every 100 updates. Artifacts contain `alignment`, `spectrum_ratio`, `target_ratio`,
and `iter`. Histories describe parameters before the corresponding update.
A full run records steps `0` through `399900`. Keep the entire MLXP run directory:
scalar metrics alone cannot produce the heatmaps. Wait for all ten runs to finish.

## Generate the four figures

From the repository root:

```bash
MPLBACKEND=Agg python peira_learning_experiments/post_runs/post_runs.py
```

The default paths are resolved from the script's location, independently of the
current working directory. The reader refreshes its index and selects completed
seed-0, 400000-step runs with the methods, regularizations, learning rates, and
initialization scales listed above. It requires exactly one matching run for each
method and scale; use a fresh sweep directory or remove duplicate runs from the
input selection before plotting.

To specify your input and output directories:

```bash
MPLBACKEND=Agg python peira_learning_experiments/post_runs/post_runs.py \
    --runs-dir /absolute/path/to/runs \
    --figures-dir /absolute/path/to/figures
```

The output directory is created automatically. Defaults write the four PNG files
to `CSRLeaning/figures`, at 300 dpi. `MPLBACKEND=Agg` supports machines without a
graphical display. Existing figures with the same names are overwritten.

Panels run from initialization scale `1e-2` at the top to `1e-6` at the bottom.
`viz.py` loads and sorts the saved artifacts and interpolates each history onto a
logarithmically spaced iteration grid. Spectral ratios are padded to the alignment
dimension. Both heatmaps use the same fixed reference: four ones followed by
zeros, shown in the red box. This reference is hard-coded to preserve the supplied
figure layout; it is not read from the logged `target_ratio`. Exact reproduction
also requires matching the original runs and numerical environment.

## Code layout

- `main.py`, `configs/`, and `scripts/toy_deterministic.sh`: MLXP launch and sweep.
- `datasets/gaussian.py`: deterministic paired toy data and ground-truth spectra.
- `training/`: encoder updates and covariance computation.
- `evaluation/`: spectrum normalization and eigenvector alignment.
- `post_runs/post_runs.py` and `post_runs/viz.py`: run selection and four heatmaps.
- `exp_types.py`, `factory.py`, and `utils/`: shared data structures, configured
  class loading, and seeding.
