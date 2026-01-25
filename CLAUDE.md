# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AITimeStepper is a physics-informed ML time stepper for N-body gravitational simulations. It trains neural networks to predict adaptive time steps (dt) for efficient N-body integration while preserving energy and momentum conservation.

## Commands

### Training
```bash
# Basic training (analytic features)
python run/runner.py train --epochs 200 --n-steps 5 --num-particles 4 --save-name my_run

# History-aware training (temporal features)
python run/runner.py train --epochs 200 --n-steps 5 --history-len 5 --feature-type delta_mag --num-particles 4 --save-name my_run

# With W&B logging
python run/runner.py train --epochs 200 --n-steps 5 --num-particles 4 --wandb --wandb-project AITimeStepper
```

### Simulation
```bash
# Analytic (no ML)
python run/runner.py simulate --num-particles 4 --steps 500

# ML-predicted dt
python run/runner.py simulate --integrator-mode ml --model-path data/<save>/model/model_epoch_XXXX.pt --num-particles 4 --steps 500

# History-aware ML
python run/runner.py simulate --integrator-mode history --model-path data/<save>/model/model_epoch_XXXX.pt --history-len 5 --feature-type delta_mag --num-particles 4 --steps 500
```

### Quick sanity check
```bash
python run/runner.py simulate --num-particles 3 --steps 200
```

### SLURM cluster jobs
```bash
sbatch run/run_ml.slurm      # GPU training
sbatch run/run_sim.slurm     # CPU simulation
```

## Architecture

### Core Components

**Config System** (`src/config.py`): Centralized dataclass for all parameters - training, simulation, loss bounds, device settings. Saved with checkpoints for reproducibility.

**Differentiable Physics** (`src/particle.py`): `ParticleTorch` class provides batched, autograd-compatible N-body state. Computes accelerations, energy, momentum with gradient flow through integration.

**Feature Extraction**:
- `src/nbody_features.py`: System-level features (CoM, distances, statistics)
- `src/history_buffer.py`: `HistoryBuffer` for temporal state tracking with feature types: `basic`, `rich`, `delta_mag`

**ML Model** (`src/structures.py`): `FullyConnectedNN` with configurable hidden layers, outputs positive dt via Softplus.

**ModelAdapter** (`src/model_adapter.py`): Abstracts feature construction (analytic vs history-aware), provides unified prediction interface.

**Loss Functions** (`src/losses.py`, `src/losses_history.py`): Physics-informed losses penalizing energy/momentum drift outside tolerance bands (`E_lower`, `E_upper`, `L_lower`, `L_upper`).

### Data Flow

```
ICs → ParticleTorch → ModelAdapter → NN → dt prediction
                         ↓
              Integration (evolve) → Loss (conservation check)
                         ↓
                  Backprop → Optimizer step
```

### Directory Structure

- `src/`: Differentiable PyTorch modules (particle, losses, model, config)
- `simulators/`: NumPy-based N-body simulator for inference
- `run/`: Entry points and SLURM scripts
- `data/`: Outputs (models saved to `data/<save_name>/model/`)
- `optuna/`: Hyperparameter sweep configuration

### Key Patterns

- **Checkpoint contract**: Models save config metadata; simulation auto-loads `history_len`, `feature_type`, `dtype` from checkpoint
- **External fields**: `PointMassTidalField` in `src/external_potentials.py` adds tidal acceleration via `particle.set_external_field()`
- **Multi-orbit training**: `--num-orbits` flag for batched training across independent systems
