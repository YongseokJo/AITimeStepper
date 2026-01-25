# Usage

## Unified runner

`run/runner.py` is the single entrypoint for both training and simulation. It uses the shared `Config` and supports analytic, ML, and history-aware ML modes.

### TOML config (train + simulate)

```bash
python run/runner.py both --config configs/example.toml
```

Example TOML:

```toml
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
# model_path optional: defaults to the final checkpoint from training
# movie = true
# movie_max_frames = 1000
```

CLI flags override TOML values (for example, `--steps 1000`).

### Train (analytic features)

```bash
python run/runner.py train \
  --epochs 200 \
  --n-steps 5 \
  --num-particles 4 \
  --save-name nbody_run
```

### Train (history-aware)

```bash
python run/runner.py train \
  --epochs 200 \
  --n-steps 5 \
  --history-len 5 \
  --feature-type delta_mag \
  --num-particles 4 \
  --save-name nbody_history_run
```

### Two-phase training controls

- `--energy-threshold`: accept/reject gate for trajectory collection
- `--steps-per-epoch`: number of validated steps per epoch (Part 1)
- `--replay-steps`, `--replay-batch-size`, `--min-replay-size`: Part 2 generalization controls
- `--duration` and `--num-orbits > 1` are accepted but warn and fall back to epoch-based single-orbit training

### Simulate (analytic)

```bash
python run/runner.py simulate \
  --num-particles 4 \
  --steps 500
```

### Simulate (ML dt)

```bash
python run/runner.py simulate \
  --integrator-mode ml \
  --model-path data/<save>/model/model_epoch_XXXX.pt \
  --num-particles 4 \
  --steps 500
```

### Simulate (history ML)

```bash
python run/runner.py simulate \
  --integrator-mode history \
  --model-path data/<save>/model/model_epoch_XXXX.pt \
  --history-len 5 \
  --feature-type delta_mag \
  --num-particles 4 \
  --steps 500
```

### Duration-based runs

```bash
# Simulate until t >= duration
python run/runner.py simulate --num-particles 4 --duration 0.05
```

### External field (tidal)

```bash
python run/runner.py simulate \
  --external-field-mass 5.0 \
  --external-field-position 10 0 0 \
  --num-particles 4 \
  --steps 200
```

### Movie generation (two-body 2D)

```bash
python run/runner.py simulate --num-particles 2 --dim 2 --steps 500 --movie
```

Options: `--movie-dir`, `--movie-max-frames`, `--movie-fps`, `--movie-dpi`.
Requires matplotlib + ffmpeg in PATH; only supports 2-body 2D runs. Training with `--movie` writes the cumulative trajectory (including warmup steps) to `nbody_movie_<save>_train.mp4`.

### Initial conditions

Random ICs (default) use:
- `--num-particles`
- `--dim`
- `--mass`
- `--pos-scale`
- `--vel-scale`

Load ICs from file:

```bash
python run/runner.py simulate --ic-path path/to/ics.npy
python run/runner.py train --ic-path path/to/ics.txt
```

## Output format

Simulation outputs JSON with residuals:

- `energy_residual = (E_final - E0) / E0`
- `momentum_residual = ||P_final - P0|| / (||P0|| + 1e-12)`
- `angular_momentum_residual` uses relative scalar change if scalar, otherwise relative L2 norm.

## Sanity check

```bash
python run/runner.py simulate --num-particles 3 --steps 200
```

## Help

```bash
python run/runner.py --help
python run/runner.py train --help
python run/runner.py simulate --help
python run/runner.py both --help
```
