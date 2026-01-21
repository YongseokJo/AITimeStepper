# Technology Stack - AITimeStepper

## Overview
AITimeStepper is a Python-based ML framework for physics-informed neural networks applied to N-body gravitational simulations. It uses PyTorch for automatic differentiation and Weights & Biases for experiment tracking.

---

## Languages & Runtime

### Primary Language
- **Python 3.10+**
  - Minimum recommended version: 3.10
  - Used throughout source, simulators, and training scripts

### Type Annotations
- Static typing via Python dataclasses and type hints
- Used in config system, particle state, and model definitions

---

## Core Frameworks & Libraries

### Machine Learning & Deep Learning
- **PyTorch 2.7.1** (with CUDA 11.8)
  - Automatic differentiation (autograd)
  - Neural network modules (nn.Module, nn.Sequential, etc.)
  - GPU acceleration via CUDA
  - Tensor operations for batched physics simulations

- **Optuna 4.6.0**
  - Hyperparameter optimization framework
  - Artifact storage via FileSystemArtifactStore
  - Supports parallel trial execution

### GPU Stack
- **NVIDIA CUDA 11.8** (configured for compute)
- **NVIDIA cuBLAS 11.11.3.6** - GPU linear algebra
- **NVIDIA cuDNN 9.1.0.70** - GPU deep learning primitives
- **NVIDIA cuFFT 10.9.0.58** - GPU FFT operations
- **NVIDIA cuSPARSE 11.7.5.86** - GPU sparse matrix operations
- **NVIDIA Triton 3.3.1** - GPU kernel generation
- **NVIDIA NCCL 2.21.5** - Multi-GPU communication
- **Torch CUDA extensions** - Custom GPU kernels (via PyTorch)

### Scientific Computing
- **NumPy 2.3.3**
  - N-body physics simulation (NumPy backend)
  - Array operations and linear algebra
  - Random number generation for initial conditions

- **Sympy 1.14.0**
  - Symbolic mathematics (for potential/acceleration calculations)

- **SciPy stack** (via dependencies)
  - Linear algebra, optimization

### Visualization & Notebooks
- **Matplotlib 3.10.7**
  - Physics simulation plots
  - Training diagnostics

- **Jupyter** ecosystem (7.1.0+ with JupyterLab 4.4.10)
  - Interactive notebook exploration
  - IPython kernel support
  - Widgets for interactive visualization

### Experiment Tracking
- **Weights & Biases (wandb) 0.23.1**
  - Run logging and visualization
  - Hyperparameter configuration saving
  - Artifact storage
  - Project organization (AITimeStepper project)
  - Email notifications on job completion

---

## Configuration & Data

### Configuration System
- **Dataclasses** (Python standard library)
  - Centralized Config class in `src/config.py`
  - Includes training parameters, loss bounds, device settings, external fields
  - CLI argument parsing via argparse
  - Checkpoint serialization

### Data Serialization
- **NumPy** (.npy format)
  - Initial conditions storage
  - Training data persistence

- **PyTorch checkpoints** (.pt format)
  - Model weights
  - Optimizer state
  - Epoch metadata
  - Config snapshots for reproducibility

- **JSON** (Python standard)
  - Simulation metrics output
  - Configuration snapshots

### YAML
- W&B metadata storage (auto-generated)
- SLURM job configuration examples

---

## Development & Build Tools

### Package Management
- **pip 25.3**
  - Dependency installation
  - Virtual environment management (via venv or pyenv)

### Job Scheduling & HPC
- **SLURM** (cluster integration)
  - GPU job submission (run_ml.slurm, run_sim.slurm, run_sim_gpu.slurm)
  - Array jobs support
  - GPU allocation (NVIDIA A40, A100 GPUs)
  - Email notifications
  - 12-hour walltime typical

### Version Control
- **Git** + **GitPython 3.1.45**
  - Commit tracking in W&B metadata
  - Repository integration

### Process Management
- **joblib** (Optuna parallel backend)
  - Parallel trial execution for hyperparameter sweeps

---

## Runtime Environment

### Development Environment
- **Virtual Environment**: pyenv (Python venv variant)
  - Located at `$HOME/pyenv/torch/bin/activate`

### Server Infrastructure
- **HPC Cluster**: Delta (likely XSEDE/ACCESS)
  - Available GPUs: NVIDIA A40x4, A100x4, A100x8
  - CPU nodes: 64-core systems
  - Scratch filesystem for I/O

### Device Abstraction
- **torch.device("auto" | "cpu" | "cuda")**
  - Auto-detection of GPU availability
  - Configurable device placement

---

## Computational Infrastructure

### PyTorch Optimizers
- **Adam** (default)
  - Learning rate: 1e-4
  - Weight decay: 1e-2

- **LBFGS** (optional second phase)
  - Line search: strong_wolfe
  - Max iterations: 500

### Logging & Debugging
- **Python logging** (via standard library)
- **W&B run tracking**
  - Epoch-wise loss logging
  - Hyperparameter snapshots
  - System metrics (GPU, CPU, memory)

### Testing & Validation
- Ad-hoc simulation sanity checks
- Physics conservation metrics (energy, momentum, angular momentum)
- Residual error calculations

---

## Key Files

- **Requirements source**: `/run/wandb/run-*/files/requirements.txt`
- **SLURM job configs**: `/run/run_ml.slurm`, `/run/run_sim.slurm`
- **Config dataclass**: `src/config.py`
- **Main runner**: `run/runner.py`
- **Optuna hyperparameter sweeps**: `optuna/main.py`

---

## Dependency Highlights

### Direct Production Dependencies
```
torch==2.7.1+cu118
numpy==2.3.3
optuna==4.6.0
wandb==0.23.1
matplotlib==3.10.7
```

### System Dependencies (from requirements)
- CUDA toolkit, cuDNN, cuBLAS (via nvidia-* packages)
- Jupyter ecosystem for exploration
- Various CLI and environment tools

### Optional/Conditional
- **wandb**: Only imported if `--wandb` flag is enabled during training

---

## Summary Table

| Category | Technology | Version | Purpose |
|----------|-----------|---------|---------|
| **Runtime** | Python | 3.10+ | Primary language |
| **ML Framework** | PyTorch | 2.7.1+cu118 | Autograd, tensor ops, GPU |
| **GPU Compute** | CUDA | 11.8 | GPU acceleration |
| **Scientific** | NumPy | 2.3.3 | N-body simulation, arrays |
| **HP Optimization** | Optuna | 4.6.0 | Hyperparameter tuning |
| **Experiment Tracking** | Weights & Biases | 0.23.1 | Run logging & tracking |
| **Visualization** | Matplotlib | 3.10.7 | Plotting & diagnostics |
| **Notebooks** | Jupyter | 7+ | Interactive exploration |
| **Job Scheduling** | SLURM | (cluster) | HPC batch jobs |
| **Version Control** | Git | (system) | Code history |
