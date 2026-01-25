# AITimeStepper

Physics-informed ML time stepper + small N-body tooling.

## Getting started

### Environment

This is a Python + PyTorch project. Create/activate an environment that has at least:

- `python` (3.10+ recommended)
- `torch`
- `numpy`
- `matplotlib`
- `wandb` (only required if you pass `--wandb`)

If you already have an environment, you can quickly sanity-check imports with:

```bash
python -c "import torch, numpy, matplotlib, wandb"
```

### Repo layout

- `src/`: differentiable particle state, losses, ML utilities
- `simulators/`: NumPy simulators / non-training utilities
- `run/`: runnable scripts (training, tests, sanity scripts)
- `jupyter_notebooks/`: exploration notebooks
- `data/`: outputs (models/logs/plots) written by runners

## Quick runs

### 1) N-body simulator sanity

```bash
python run/runner.py simulate --num-particles 3 --steps 200
```

### 2) Train + simulate from a TOML config

```bash
python run/runner.py both --config configs/example.toml
```

### 3) Run Optuna sweeps

See `optuna/main.py` and `run/run_optuna.sh` for the current sweep entrypoints.

## TOML workflow

Use `--config` to pass a TOML file with shared defaults plus `[train]` and `[simulate]` sections.
CLI flags can still override values from TOML.

```toml
# shared defaults
save_name = "nbody_run"
num_particles = 4
history_len = 5
feature_type = "delta_mag"

[train]
epochs = 200
n_steps = 5
energy_threshold = 2e-4
steps_per_epoch = 2
wandb = true

[simulate]
steps = 500
# model_path optional: defaults to latest checkpoint from training
# movie = true
# movie_max_frames = 1000
```

## Unified runner (single entrypoint)

Use `run/runner.py` for both training and simulation with the shared `Config`.

### Train (N-body)

```bash
python run/runner.py train --epochs 200 --n-steps 5 --num-particles 4 --save-name nbody_run
```

### Train (history-aware)

```bash
python run/runner.py train --epochs 200 --n-steps 5 --history-len 5 --feature-type delta_mag --num-particles 4 --save-name nbody_history_run
```

### Two-phase training notes

- Training uses an accept/reject loop that enforces `--energy-threshold` on every step.
- Use `--steps-per-epoch` to control how many validated steps are collected per epoch.
- `--duration` and `--num-orbits > 1` are accepted but currently warn and fall back to single-orbit epoch-based training.

### Simulate (analytic)

```bash
python run/runner.py simulate --num-particles 4 --steps 500
```

### Simulate (ML dt)

```bash
python run/runner.py simulate --integrator-mode ml --model-path data/<save>/model/model_epoch_XXXX.pt --num-particles 4 --steps 500
```

### Simulate (history ML)

```bash
python run/runner.py simulate --integrator-mode history --model-path data/<save>/model/model_epoch_XXXX.pt --history-len 5 --feature-type delta_mag --num-particles 4 --steps 500
```

### Duration-based runs

```bash
python run/runner.py simulate --num-particles 4 --duration 0.05
```

### External field

```bash
python run/runner.py simulate --external-field-mass 5.0 --external-field-position 10 0 0 --num-particles 4 --steps 200
```

### Movie generation (two-body 2D)

```bash
python run/runner.py simulate --num-particles 2 --dim 2 --steps 500 --movie
```

Options: `--movie-dir`, `--movie-max-frames`, `--movie-fps`, `--movie-dpi`.
Requires matplotlib + ffmpeg in PATH; only supports 2-body 2D runs. Training with `--movie` writes the cumulative trajectory (including warmup steps) to `nbody_movie_<save>_train.mp4`.

## Analytic tidal potential (external field)

This repo supports adding an analytic **external gravitational field** on top of the internal N-body forces.

### What’s implemented

- `PointMassTidalField` in `src/external_potentials.py`
- Adds a tidal (differential) acceleration from a distant point-mass perturber at position $R$:

$$
\mathbf{a}(\mathbf{x}) = G M\left[\frac{\mathbf{R}-\mathbf{x}}{|\mathbf{R}-\mathbf{x}|^3} - \frac{\mathbf{R}}{|\mathbf{R}|^3}\right]
$$

The uniform acceleration of the origin is subtracted (so this is a true “tide”).

The matching potential $\Phi$ is also implemented so `ParticleTorch.total_energy(...)` and `total_energy_batch(...)` can include the external potential energy:

$$
\Phi(\mathbf{x}) = -G M\left[\frac{1}{|\mathbf{R}-\mathbf{x}|} - \frac{1}{|\mathbf{R}|} - \frac{\mathbf{x}\cdot\mathbf{R}}{|\mathbf{R}|^3}\right]
$$

### PyTorch usage (training / differentiable)

Attach an external field to a `ParticleTorch` state:

```python
from src.external_potentials import PointMassTidalField

tide = PointMassTidalField(M=5.0, R0=[10.0, 0.0], G=1.0)
particle.set_external_field(tide)
```

Once attached:
- `ParticleTorch.get_acceleration()` returns internal N-body acceleration + external acceleration.
- `ParticleTorch.total_energy()` / `total_energy_batch()` include the external potential energy.

### NumPy simulator usage

The NumPy N-body simulator supports an optional `external_acceleration(positions, time)` callback:

```python
from simulators.nbody_simulator import evolve_particles

def ext_acc(pos, t):
    # pos: (N, D) numpy array
    # return: (N, D) numpy array
    ...

evolve_particles(particles, external_acceleration=ext_acc)
```

### Quick sanity run

Use the unified runner:

```bash
python run/runner.py simulate --external-field-mass 5.0 --external-field-position 10 0 0 --num-particles 4 --steps 200
```
