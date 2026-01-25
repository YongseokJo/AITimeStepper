# AITimeStepper Context for Gemini

## Project Overview

**AITimeStepper** is a physics-informed machine learning framework designed to learn adaptive time-stepping (`dt`) for N-body gravitational simulations. By training neural networks to predict optimal time steps, it aims to maintain high accuracy (energy/momentum conservation) while maximizing simulation efficiency, potentially outperforming traditional adaptive integrators in specific regimes.

**Core Goal:** Replace heuristic based adaptive time-stepping with a learned policy that respects conservation laws.

## Tech Stack & Environment

*   **Language:** Python 3.10+
*   **Deep Learning:** PyTorch 2.7.1+ (CUDA 11.8 supported)
*   **Physics/Numerics:** NumPy 2.3.3 (inference), SymPy (symbolic)
*   **Experiment Tracking:** Weights & Biases (wandb) 0.23.1
*   **Optimization:** Optuna 4.6.0 (hyperparameter sweeps)
*   **Visualization:** Matplotlib, Jupyter, FFMPEG (for movies)
*   **Job Scheduler:** SLURM (for HPC runs)

### Setup
The project relies on standard Python environment tools.
```bash
# Recommended setup
python -m venv venv
source venv/bin/activate
pip install torch numpy matplotlib wandb optuna
```

## Key Entry Points

The project uses a **Unified Runner** pattern. The primary entry point is `run/runner.py`.

### 1. Unified Runner (`run/runner.py`)
Handles both training and simulation. Accepts CLI arguments and TOML configuration files.

*   **Train Mode:** Trains the ML model to predict `dt`.
    ```bash
    python run/runner.py train --num-particles 4 --epochs 200 --save-name my_run
    ```
*   **Simulate Mode:** Runs N-body simulation (Analytic or ML-driven).
    ```bash
    # Analytic (fixed dt)
    python run/runner.py simulate --num-particles 4 --steps 500
    # ML-driven (uses trained model)
    python run/runner.py simulate --integrator-mode ml --model-path data/<run>/model.pt ...
    ```
*   **Both:** Trains then immediately simulates.
    ```bash
    python run/runner.py both --config configs/example.toml
    ```

### 2. Configuration (`src/config.py`)
Configuration is centralized in a `Config` dataclass.
*   **TOML:** Preferred for reproducible experiments (`configs/example.toml`).
*   **CLI:** Overrides TOML values (e.g., `--epochs 100`).
*   **Checkpointing:** The `Config` object is serialized with model checkpoints to ensure inference parameters match training.

## Architecture & Code Structure

The system is designed with layered abstractions:

1.  **Configuration (`src/config.py`)**: Central source of truth. Validates parameters.
2.  **Physics Engine (`src/particle.py`)**: `ParticleTorch` class. Differentiable physics state (position, velocity, mass). Supports Batched operations `(B, N, D)`.
3.  **Feature Extraction (`src/nbody_features.py`, `src/history_buffer.py`)**:
    *   `ParticleTorch.system_features()`: Extracts analytic features (energies, derivatives).
    *   `HistoryBuffer`: Manages temporal history for sequence-based models.
4.  **Model Adapter (`src/model_adapter.py`)**: Bridges the Physics engine and the Neural Network. Handles feature construction and history management seamlessly.
5.  **Model (`src/structures.py`)**: `FullyConnectedNN`. Simple MLP predicting `dt` (and auxiliary params).
6.  **Losses (`src/losses.py`)**: Physics-Informed Losses.
    *   Simulates `n_steps` forward using predicted `dt`.
    *   Penalizes deviations in Energy (`dE`) and Momentum (`dL`).
7.  **Inference (`simulators/nbody_simulator.py`)**: Fast, pure-NumPy integrator for production simulation (no autograd overhead).

## Development Conventions

*   **Type Hinting:** Strict usage of modern Python type hints (`|` unions, `Optional`, `List`).
*   **Tensor Shapes:**
    *   `N`: Number of particles.
    *   `D`: Dimensionality (usually 2 or 3).
    *   `B`: Batch size (for parallel training orbits).
    *   Standard shape: `(B, N, D)` or `(N, D)`.
*   **Coding Style:**
    *   **Classes**: `PascalCase` (e.g., `ParticleTorch`, `ModelAdapter`).
    *   **Functions/Vars**: `snake_case` (e.g., `get_acceleration`, `loss_fn_batch`).
    *   **Constants**: `G=1.0` is standard.
*   **Safety:**
    *   Use `torch.no_grad()` for inference.
    *   Use `clone_detached()` when storing states to break computation graphs.
    *   Use `softening` parameter to prevent gravitational singularities.

## Common Development Tasks

**Running Tests:**
```bash
# Ad-hoc sanity check
python run/runner.py simulate --num-particles 3 --steps 10
# Full simulation test
python run/runner.py both --config configs/example.toml
```

**Creating a New Feature:**
1.  Add logic to `src/nbody_features.py`.
2.  Register feature name in `src/config.py`.
3.  Update `src/model_adapter.py` to use the new feature.

**Adding a Loss Function:**
1.  Implement in `src/losses.py`.
2.  Add to training loop in `run/runner.py`.

## Directory Layout
*   `src/`: Core logic (Physics, ML, Config).
*   `simulators/`: NumPy-based simulation engines.
*   `run/`: Executable scripts (`runner.py`, SLURM scripts).
*   `configs/`: TOML configuration files.
*   `data/`: (Generated) Model checkpoints, logs, and plots.
*   `tests/`: Unit and integration tests.
*   `.planning/`: Project documentation and architectural plans.
