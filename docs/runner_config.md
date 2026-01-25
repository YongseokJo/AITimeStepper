# Runner configuration (TOML + CLI)

`run/runner.py` supports three modes: `train`, `simulate`, and `both` (train then simulate).
Configuration can come from:
1. **TOML** via `--config path.toml`
2. **CLI flags** (override TOML values)

For `both`, the same TOML file provides `[train]` and `[simulate]` sections. If `model_path` is omitted in `[simulate]`, it defaults to the final checkpoint from training: `data/<save_name>/model/model_epoch_<epochs-1>.pt`.

## Basic usage

```bash
# Train only
python run/runner.py train --config configs/example.toml

# Simulate only
python run/runner.py simulate --config configs/example.toml

# Train + simulate in one job
python run/runner.py both --config configs/example.toml
```

## TOML structure

Top-level keys act as shared defaults. Use `[train]` and `[simulate]` to override per-mode settings.

```toml
save_name = "nbody_run"
num_particles = 4
history_len = 5
feature_type = "delta_mag"

[train]
epochs = 200
n_steps = 5
steps_per_epoch = 2
energy_threshold = 2e-4
max_retrain_steps = 1000
wandb = true

[simulate]
steps = 500
# model_path optional: defaults to the final checkpoint from training
# movie = true
# movie_max_frames = 1000
```

## CLI overrides

CLI flags always override TOML values. Example:

```bash
python run/runner.py both --config configs/example.toml --steps 1000 --energy-threshold 1e-4
```

## Configuration reference

### Training / optimization

| TOML key | CLI flag | Description | Default |
| --- | --- | --- | --- |
| `optimizer` | `--optimizer` | Optimizer type | `adam` |
| `epochs` | `--epochs`, `-n` | Training epochs | `1000` |
| `lr` | `--lr` | Learning rate | `1e-4` |
| `weight_decay` | `--weight-decay` | Weight decay | `1e-2` |
| `momentum` | `--momentum` | SGD momentum | `0.9` |
| `n_steps` | `--n-steps` | Integration steps per loss eval | `10` |
| `dt_bound` | `--dt-bound` | dt bound (loss heuristics) | `1e-8` |
| `rel_loss_bound` | `--rel-loss-bound` | Relative loss bound | `1e-5` |
| `energy_threshold` | `--energy-threshold` | Accept/reject energy threshold | `2e-4` |
| `steps_per_epoch` | `--steps-per-epoch` | Accepted steps per epoch | `1` |
| `max_retrain_steps` | `--max-retrain-steps` | Max retrain iterations per step | `1000` |
| `replay_steps` | `--replay-steps` | Max replay optimization steps | `1000` |
| `replay_batch_size` | `--replay-batch-size` | Replay batch size | `512` |
| `min_replay_size` | `--min-replay-size` | Min replay size before training | `2` |

### Loss bounds

| TOML key | CLI flag | Description | Default |
| --- | --- | --- | --- |
| `E_lower` | `--E_lower` | Lower energy bound | `1e-6` |
| `E_upper` | `--E_upper` | Upper energy bound | `1e-4` |
| `L_lower` | `--L_lower` | Lower angular momentum bound | `1e-4` |
| `L_upper` | `--L_upper` | Upper angular momentum bound | `1e-2` |

### Feature / history

| TOML key | CLI flag | Description | Default |
| --- | --- | --- | --- |
| `history_len` | `--history-len` | History length | `0` |
| `feature_type` | `--feature-type` | Feature type (`basic`, `rich`, `delta_mag`) | `delta_mag` |

### Simulation / integrator

| TOML key | CLI flag | Description | Default |
| --- | --- | --- | --- |
| `integrator_mode` | `--integrator-mode` | `analytic`, `ml`, or `history` | `analytic` |
| `dt` | `--dt` | Fixed dt (analytic) | `-1.0` |
| `steps` | `--steps` | Steps to simulate | `-1` |
| `duration` | `--duration` | Stop when time >= duration | `None` |
| `Nperiod` | `--Nperiod` | Period count | `10` |
| `eps` | `--eps` | dt update constant | `0.1` |
| `model_path` | `--model-path` | Path to checkpoint | `None` |
| `num_particles` | `--num-particles` | Particle count | `2` |
| `dim` | `--dim` | Spatial dimension | `2` |
| `mass` | `--mass` | Particle mass (random ICs) | `1.0` |
| `pos_scale` | `--pos-scale` | Random IC position scale | `0.1` |
| `vel_scale` | `--vel-scale` | Random IC velocity scale | `1.0` |

### External field (optional)

| TOML key | CLI flag | Description | Default |
| --- | --- | --- | --- |
| `external_field_mass` | `--external-field-mass` | External field mass | `None` |
| `external_field_position` | `--external-field-position` | External field position (x y z) | `None` |

### Multi-orbit sampling (training and simulation)

| TOML key | CLI flag | Description | Default |
| --- | --- | --- | --- |
| `num_orbits` | `--num-orbits` | Orbit batch count | `8` |
| `stop_at_period` | `--stop-at-period` | Stop collection when period is reached | `false` |
| `e_min` | `--e-min` | Min eccentricity | `0.6` |
| `e_max` | `--e-max` | Max eccentricity | `0.95` |
| `a_min` | `--a-min` | Min semi-major axis | `0.8` |
| `a_max` | `--a-max` | Max semi-major axis | `1.2` |

### Device / dtype

| TOML key | CLI flag | Description | Default |
| --- | --- | --- | --- |
| `device` | `--device` | `auto`, `cpu`, `cuda` | `auto` |
| `dtype` | `--dtype` | `float32` or `float64` | `float64` |
| `tf32` | `--tf32` | Enable TF32 | `false` |
| `compile` | `--compile` | Use torch.compile | `false` |
| `detect_anomaly` | `--detect-anomaly` | Autograd anomaly detection | `false` |

### Logging / misc

| TOML key | CLI flag | Description | Default |
| --- | --- | --- | --- |
| `save_name` | `--save-name`, `-s` | Output name under data/ | `None` |
| `seed` | `--seed` | RNG seed | `None` |
| `debug` | `--debug` | Enable debug logs | `false` |
| `debug_every` | `--debug-every` | Debug print interval | `1` |
| `debug_replay_every` | `--debug-replay-every` | Debug replay interval | `10` |

### Movie generation (simulate + train)

| TOML key | CLI flag | Description | Default |
| --- | --- | --- | --- |
| `movie` | `--movie` | Save MP4 movie (training uses cumulative trajectory, including warmup steps) | `false` |
| `movie_dir` | `--movie-dir` | Output directory | `None` (defaults to `data/movie`) |
| `movie_max_frames` | `--movie-max-frames` | Max frames (downsample) | `1000` |
| `movie_fps` | `--movie-fps` | Frames per second | `30` |
| `movie_dpi` | `--movie-dpi` | Render DPI | `120` |

### Initial conditions

These are not part of `Config`, but are accepted by the runner as extra fields:

| Use case | CLI flag | TOML key | Description |
| --- | --- | --- | --- |
| Train ICs | `--ic-path` (train) / `--train-ic-path` (both) | `ic_path` (under `[train]` or top-level) | Path to `.npy` or `.txt` ICs |
| Sim ICs | `--ic-path` (simulate) / `--sim-ic-path` (both) | `ic_path` (under `[simulate]` or top-level) | Path to `.npy` or `.txt` ICs |
| Sim ICs (from checkpoint) | `--use-train-ic` | `use_train_ic` (under `[simulate]` or top-level) | Use training ICs stored in the model checkpoint |

### Simulation extras

| Use case | CLI flag | TOML key | Description |
| --- | --- | --- | --- |
| Multi-orbit simulation | `--multi-orbit-sim` | `multi_orbit_sim` (under `[simulate]` or top-level) | Run `num_orbits` independent 2-body orbits sampled from `e_min/e_max` and `a_min/a_max` (optionally include training ICs) |

## Mode-specific notes

- `both` runs training then simulation using the same TOML file.
- If `model_path` is omitted for simulation in `both`, it defaults to the final checkpoint from training.
- `duration` is accepted but will warn and fall back to epoch-based training.
- `num_orbits > 1` is supported in both training and simulation.
- Multi-orbit simulation supports 2-body systems in >=2D and uses `num_orbits`, `e_min/e_max`, and `a_min/a_max` to sample extra orbits.
- Movie generation only supports two-body 2D runs and requires matplotlib + ffmpeg.
- Training movies are written as `nbody_movie_<save>_train.mp4` using the cumulative trajectory (including warmup steps).
