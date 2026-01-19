# AITimeStepper

Physics-informed ML time stepper + small N-body tooling.

## Getting started

### Environment

This is a Python + PyTorch project. Create/activate an environment that has at least:

- `python` (3.10+ recommended)
- `torch`
- `numpy`
- `matplotlib`
- `wandb` (only required for the `*_wandb.py` runners)

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

### 2) Train the history-aware time-stepper (W&B)

```bash
python run/ML_history_wandb.py --epochs 1000 --n-steps 10 --history-len 3 --feature-type basic
```

Useful flags:

- `--device auto|cpu|cuda`
- `--E_lower`, `--E_upper` (energy drift band)
- `--L_lower`, `--L_upper` (angular momentum drift band)

If training appears to stall, run with debug timing:

```bash
python run/ML_history_wandb.py --debug --debug-every 1 --debug-replay-every 10
```

### 3) Run Optuna sweeps

See `optuna/main.py` and `run/run_optuna.sh` for the current sweep entrypoints.

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
python run/runner.py train --epochs 1000 --duration 60
```

### External field

```bash
python run/runner.py simulate --external-field-mass 5.0 --external-field-position 10 0 0 --num-particles 4 --steps 200
```

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

There’s a tiny demo script:

```bash
python run/tidal_sanity.py
```
