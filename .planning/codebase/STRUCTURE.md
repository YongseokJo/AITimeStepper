# STRUCTURE.md - AITimeStepper Directory Layout

## Root Directory Structure

```
AITimeStepper/
├── run/                          # Entry points and SLURM scripts
├── src/                          # Core library modules (PyTorch)
├── simulators/                   # NumPy-based simulation engine
├── optuna/                       # Hyperparameter optimization
├── data/                         # Output directory (symlink to /work/hdd/...)
├── docs/                         # Documentation
├── jupyter_notebooks/            # Analysis notebooks
├── .planning/                    # Planning documents and intelligence
│   └── codebase/                # Architecture and structure documentation
├── README.md                     # Project overview
├── CLAUDE.md                     # Developer guidance for Claude
├── USAGE.md                      # Usage instructions
├── AGENTS.md                     # Agent instructions
└── Notes.md                      # Research notes
```

---

## Core Modules (`src/`)

### Configuration & State

**`src/config.py`** (274 lines)
- **`Config`**: Central dataclass for all parameters
  - Training: epochs, lr, weight_decay, n_steps, optimizer settings
  - Bounds: E_lower/E_upper, L_lower/L_upper (energy/momentum conservation bounds)
  - Features: history_len, feature_type (basic/rich/delta_mag)
  - Physics: dt, steps, duration, eps, softening
  - Device: device (auto/cpu/cuda), dtype (float32/float64), tf32, compile flags
  - Multi-orbit: num_orbits, e_min/e_max, a_min/a_max (for batched training)
  - External fields: external_field_mass, external_field_position (for tidal fields)
  - Checkpointing: model_path, save_name, seed
- **Key methods**:
  - `add_cli_args()`: Dynamically generates argparse arguments with filtering
  - `from_dict()`/`to_dict()`: Serialization for checkpoint contract
  - `resolve_device()`: Auto-detect CUDA availability
  - `resolve_dtype()`: Map string dtype to torch.dtype
  - `validate()`: Check consistency (e.g., history_len > 0 requires feature_type)
  - `as_wandb_dict()`: Format for Weights & Biases logging

**Key invariants**:
- Saved with every checkpoint; auto-loaded on inference
- Ensures reproducibility across runs and devices
- Extra dict for unknown/custom fields

---

### Differentiable Physics Engine

**`src/particle.py`** (572 lines)
- **`ParticleTorch`**: Batched N-body state with autograd support
  - **State tensors**: `position`, `velocity`, `acceleration`, `mass`, `dt`
  - **Shapes**: Single (N, D) or batched (B, N, D)
  - **Softening**: Numerical stability for close encounters
  - **External fields**: Optional `external_field` (ExternalField subclass)
  - **Tracking**: `current_time`, `period` (bookkeeping, not part of autograd graph)

- **Constructors**:
  - `__init__()`: For non-training use; wraps inputs in new tensors
  - `from_tensors()`: For training; stores tensors as-is to preserve autograd
  - `clone_detached()`: Creates fresh copy with detached tensors

- **Physics methods**:
  - `get_acceleration()`: Softened Newtonian gravity with pairwise distances
    - Avoids self-interaction via masked_fill(eye, inf)
    - Supports external fields via `external_field.acceleration()`
    - Fully differentiable
  - `evolve()`: Single leapfrog step (symplectic, 2nd order)
    - Kick-drift-kick pattern for energy conservation
    - In-place updates for efficiency
  - `evolve_batch()`: Handles both single and batched cases
  - `total_energy()`: Kinetic + Potential (differentiable)
  - `total_energy_batch()`: Batched energy computation
  - `kinetic_energy()`: Per-particle or summed

- **Feature extraction**:
  - `system_features(mode='basic'|'rich')`: Fixed-size permutation-invariant vectors
  - `features()`/`_get_batch_()`: Aliases for backward compatibility
  - Calls `nbody_features.system_features()` under the hood

- **State management**:
  - `reset_state()`: Reuse object with new tensors
  - `update_dt()`: Change time step
  - `set_external_field()`: Attach analytical potential

- **Utility functions** (module-level):
  - `make_particle()`: Convert numpy IC array (N, 1+2D) → ParticleTorch
    - Format: [mass, x..., y..., vx..., vy...]
  - `stack_particles()`: Combine multiple ParticleTorch → batched ParticleTorch
  - `generate_IC()`: Generate binary orbital IC for testing
  - `stack_particles()`: Batch multiple single systems

---

### Feature Extraction

**`src/nbody_features.py`** (100+ lines)
- **`system_features()`**: Compute fixed-size feature vector
  - **basic mode**: COM, mean distance, distance std, velocity stats, acceleration stats
  - **rich mode**: Extended set including higher-order moments
  - **Permutation-invariant**: Aggregate statistics over particles
  - **Supports batching**: Single (N, D) → scalar or (B, N, D) → (B, F)

- **`_compute_acceleration()`**: Helper for acceleration-based features

- **Feature dimensions**:
  - basic: ~6-10 scalars
  - rich: ~12-20 scalars
  - Can vary by N (number of particles)

**Contract**: All features are order-independent and scale-invariant where applicable.

---

### Temporal State Tracking

**`src/history_buffer.py`** (200+ lines)
- **`HistoryBuffer`**: Maintains circular buffer of past particle states
  - **Capacity**: `history_len` (deque with maxlen)
  - **Storage**: `_HistoryState` (frozen dataclass with position, velocity, mass, dt, softening)
  - **Feature types**: basic, rich, delta_mag

- **Key methods**:
  - `push()`: Add detached clone of current state
  - `features_for()`: Build concatenated feature vector from history + current
    - Single system: 1D vector of length (history_len+1) × F
    - Left-pads with oldest state if insufficient history
  - `features_for_histories()`: Parallel feature extraction for multiple systems
    - Input: List of HistoryBuffer + batch ParticleTorch
    - Output: Batched features (B, F)
  - `reset()`: Clear buffer

- **Feature types**:
  - **basic/rich**: Concatenate system features from each time step
  - **delta_mag**: Changes in |Δposition|, |Δvelocity|, |Δacceleration| + dt history
    - Captures temporal dynamics for dt prediction

- **Padding strategy**: If fewer than history_len states, repeats oldest to fill buffer
  - Ensures consistent input dimension for NN model

---

### Model Integration Layer

**`src/model_adapter.py`** (115 lines)
- **`ModelAdapter`**: Abstraction over feature extraction and model prediction
  - **Initialization**: Takes Config, device, dtype
  - **Auto-creates HistoryBuffer** if `config.history_len > 0`
  - **Handles both analytic and history-based features**

- **Key methods**:
  - `build_feature_tensor()`: Construct model input
    - History-enabled: Calls `history_buffer.features_for()` or batch version
    - Analytic: Calls `state.system_features(mode=...)`
  - `input_dim_from_state()`: Probe model input dimension from state
    - Used during model initialization
  - `predict_dt()`: Run model in no_grad mode, return dt prediction
    - Handles device/dtype transfers
    - Ensures positive output via eps offset
  - `update_history()`: Push state to history buffer if enabled
    - Optional token to deduplicate updates

- **Properties**:
  - `history_enabled`: Boolean from config
  - `feature_mode()`: Maps config.feature_type to extraction mode

**Design**: Runner code calls adapter methods; adapter internally switches between analytic/history based on config.

---

### Neural Network Model

**`src/structures.py`** (80+ lines)
- **`SimpleNN`**: Toy single-hidden-layer network (legacy)

- **`FullyConnectedNN`**: Main model architecture
  - **Architecture**: Configurable hidden layers with activation + dropout
  - **Input**: Feature vector of size input_dim
  - **Output**: 2 values (dt_raw, aux_raw)
  - **Activation options**: relu, tanh, sigmoid, silu
  - **Dropout**: Per-layer dropout for regularization
  - **Output layer**: Softplus if `output_positive=True` for ensuring positive dt

- **Standard configuration** (from runner.py):
  ```python
  FullyConnectedNN(
      input_dim=adapter.input_dim_from_state(...),
      output_dim=2,
      hidden_dims=[200, 1000, 1000, 200],
      activation='tanh',
      dropout=0.2,
      output_positive=True
  )
  ```

- **Forward pass**: Feature tensor → Linear layers with activation → Softplus → [dt, aux]

---

### Physics-Informed Loss Functions

**`src/losses.py`** (200+ lines)
- **Analytic feature loss functions**:
  - **`loss_fn_batch()`**: Core loss for single or batched systems
    - Clone particle (detach graph)
    - Extract features
    - Run model → dt, auxiliary parameter
    - Integrate n_steps with leapfrog
    - Compute energy/momentum changes
    - Penalties for violating bounds

- **Loss components**:
  - **Energy conservation**: `|ΔE/E|` penalized if outside [E_lower, E_upper]
  - **Angular momentum**: `|ΔL/L|` penalized if outside [L_lower, L_upper]
  - **dt constraint**: Softness on predicted dt value
  - **Prediction matching**: Model predicts energy change; compare to observed

- **Return**: (loss_scalar, logs_dict) where logs contain diagnostics

**`src/losses_history.py`** (200+ lines)
- **Temporal feature loss functions**:
  - **`loss_fn_batch_history()`**: Single system with history buffer
  - **`loss_fn_batch_history_batch()`**: Multiple systems with histories
  - Similar structure to analytic losses but feed history features to model
  - Push state to history buffer after integration

- **Supports**: Multi-orbit training with per-orbit history tracking

---

### Checkpointing & Serialization

**`src/checkpoint.py`** (100 lines)
- **`save_checkpoint()`**: Persist model + metadata
  - Arguments: path, model, optimizer, epoch, loss, logs, config, extra
  - Creates parent directories
  - Saves:
    - `model_state_dict`: Model weights
    - `optimizer_state_dict`: Optimizer state (for resuming training)
    - `epoch`, `loss`, `logs`: Training metadata
    - **`config`**: Full Config object (as dict) → enables auto-loading on inference
    - **`history_len`, `feature_type`, `dtype`**: Explicit metadata for inference

- **`load_checkpoint()`**: Load dict from checkpoint file
  - Takes map_location for device transfer

- **`load_model_state()`**: Extract and load model weights into provided model object
  - Handles old checkpoint formats (model_state vs model_state_dict)

- **`load_config_from_checkpoint()`**: Extract Config object from checkpoint
  - Returns None if not present (backward compatibility)

**Contract**: Any checkpoint can be used for inference by loading config metadata.

---

### Integration Utilities

**`src/integrator.py`** (21 lines)
- **`simple_integrator()`**: Toy free-drift example
- Mostly placeholder; actual integration is in ParticleTorch.evolve()

---

### External Field Support

**`src/external_potentials.py`** (100+ lines)
- **`ExternalField`**: Abstract base class
  - `acceleration(pos, time)`: Compute external acceleration at position(s)
  - `potential(pos, time)`: Compute potential energy

- **`PointMassTidalField`**: Tidal field implementation
  - Represents external point mass at fixed position
  - Computes tidal acceleration (differential gravity)
  - Optional time-dependent variations

---

## Simulation Engine (`simulators/`)

**`simulators/nbody_simulator.py`** (300+ lines)
- **Inference-time NumPy-based N-body integrator**
  - Used during simulation mode, not training

- **`generate_random_ic()`**: Random initial conditions
  - num_particles, dim, mass (scalar or array)
  - pos_scale, vel_scale
  - Centers on COM

- **`compute_accelerations()`**: Vectorized pairwise N-body gravity
  - Supports external acceleration callback
  - Softening for stability

- **`evolve_step()`**: Single leapfrog step

- **`evolve_particles()`**: Integration loop with fixed dt
  - Tracks energy, momentum, angular momentum

- **`evolve_particles_ml()`**: Integration with ML-predicted dt
  - Calls model via adapter
  - Updates history buffer if in history mode
  - Logs predictions and diagnostics

- **`total_energy()`, `total_momentum()`, `total_angular_momentum()`**: Observables

**`simulators/particle.py`** (150+ lines)
- **`Particle`**: NumPy representation for inference
  - Position, velocity, mass, softening
  - Can attach ML model for dt prediction

- **Utility functions**:
  - `predict_dt_from_model_system()`: Single system prediction
  - `predict_dt_from_history_model_system()`: History-aware prediction

---

## Runner & Entry Points (`run/`)

**`run/runner.py`** (429 lines)
- **Main orchestrator for training and simulation**

- **`run_training()`**: Training loop
  - Resolve device/dtype
  - Create Config and ModelAdapter
  - Initialize FullyConnectedNN model
  - Setup Adam optimizer
  - For each epoch:
    - Compute loss (analytic or history-based)
    - Backward pass
    - Optimizer step
    - Log to W&B if enabled
    - Save checkpoint every 10 epochs
  - Support for multi-orbit batching with num_orbits > 1

- **`run_simulation()`**: Simulation loop
  - Load ICs (random or from file)
  - Load model if ml/history mode
  - Auto-load history_len, feature_type, dtype from checkpoint config
  - Evolve particles with analytic or ML dt
  - Track conservation metrics
  - Output JSON with energy/momentum residuals

- **`build_parser()`**: Construct argparse for train/simulate subcommands
  - Train: Training params + history + bounds
  - Simulate: Simulation params + history + device

- **Mode subcommands**:
  - `train`: Train new model
  - `simulate`: Run simulation (optionally with loaded model)

- **Helper functions**:
  - `_load_ic()`: Load numpy IC from .npy or .txt
  - `_make_sim_particles()`: Convert numpy IC to SimParticle list
  - `_external_accel_from_config()`: Create external field callback

**`run/run_ml.slurm`** (SLURM script for training on GPU)
**`run/run_sim.slurm`** (SLURM script for simulation on CPU)

**`run/legacy/`**: Old experimental scripts (marked for deletion)
- `ML_history_wandb.py`, `ML_history_multi_wandb.py`: Previous training loops
- `integration_sanity.py`, `tidal_sanity.py`: Validation scripts

---

## Hyperparameter Optimization (`optuna/`)

**`optuna/main.py`**
- Optuna study configuration for automated hyperparameter tuning
- Searches over learning rate, layer sizes, dropout, loss bounds
- Minimizes validation error on held-out test set

---

## Data & Outputs

**`data/`** (symlink to `/work/hdd/bfpt/gkerex/AITimeStepper`)
- **`data/<save_name>/`**: Directory per training run
  - **`model/`**: Saved checkpoints
    - `model_epoch_XXXX.pt`: Checkpoint at epoch XXXX
  - Logs, tensorboard, W&B artifacts

---

## Documentation

**`docs/`**
- Additional documentation (integration notes, derivations, etc.)

**`README.md`**
- Project overview, installation, quickstart

**`CLAUDE.md`**
- Developer guidance for Claude AI
- Commands for training and simulation
- Architecture overview
- Quick sanity checks

**`USAGE.md`**
- Detailed usage instructions
- CLI flags and examples
- Troubleshooting

**`AGENTS.md`**
- Instructions for running multi-agent analysis

**`.planning/codebase/`**
- **`ARCHITECTURE.md`**: This document - system design, layers, data flow
- **`STRUCTURE.md`**: Directory layout, file organization, naming conventions

---

## Naming Conventions

### Classes
- **Torch-based**: `ParticleTorch`, `FullyConnectedNN`, `ModelAdapter` (PascalCase)
- **NumPy-based**: `Particle` (simulator version, PascalCase)
- **Config**: `Config` (dataclass, PascalCase)
- **Buffers**: `HistoryBuffer` (PascalCase)

### Functions
- **Physics**: `evolve()`, `get_acceleration()`, `total_energy()` (snake_case)
- **Features**: `system_features()`, `build_feature_tensor()` (snake_case)
- **Loss**: `loss_fn_batch()`, `loss_fn_batch_history()` (snake_case, `_fn_` infix)
- **Utilities**: `make_particle()`, `stack_particles()`, `generate_random_ic()` (snake_case)

### Variables
- **State**: `position`, `velocity`, `acceleration`, `mass` (geometric names)
- **Time**: `dt`, `current_time`, `period` (physics notation)
- **Parameters**: `lr` (learning rate), `wd` (weight decay), `n_steps` (step count)
- **Bounds**: `E_lower`, `E_upper`, `L_lower`, `L_upper` (physics notation)

### Files
- **Core modules**: `particle.py`, `config.py`, `model_adapter.py` (lowercase, single responsibility)
- **Collections**: `losses.py`, `structures.py` (plural when multiple classes)
- **Legacy**: `losses.py` vs `losses_history.py` (feature-based suffix)
- **Entry points**: `runner.py`, `main.py` (standard naming)

### Directories
- **Python packages**: `src/`, `simulators/` (lowercase, plural)
- **Data**: `data/`, `optuna/` (lowercase)
- **Execution**: `run/` (verb-based, scripts and SLURM)
- **Planning**: `.planning/codebase/` (dotfile, descriptive)

---

## Key File Relationships

### Training Pipeline
```
runner.py ---> Config ---> ModelAdapter ---> FullyConnectedNN
                 |             |
                 v             v
            checkpoint.py   particle.py
                             |
                             v
                     loss_fn_batch()
                             |
        +--------------------+----+--------------------+
        |                    |    |                    |
    nbody_features  history_buffer  losses.py    losses_history.py
```

### Inference Pipeline
```
runner.py ---> nbody_simulator.py ---> checkpoint.py
                |                      |
                v                      v
         evolve_particles_ml   load_config_from_checkpoint()
                |
                v
          ModelAdapter
                |
                v
          FullyConnectedNN
```

---

## Configuration to Code Mapping

| Config Parameter | File(s) Used | Purpose |
|------------------|--------------|---------|
| `history_len` | ModelAdapter, HistoryBuffer, runner | Temporal window size |
| `feature_type` | ModelAdapter, nbody_features, HistoryBuffer | Feature extraction mode |
| `n_steps` | losses.py, runner | Integration steps per loss eval |
| `E_lower/E_upper` | losses.py, runner | Energy conservation bounds |
| `lr, weight_decay` | runner (optimizer setup) | Adam parameters |
| `epochs` | runner (training loop) | Training duration |
| `device, dtype` | Config resolution, ParticleTorch | Computation platform |
| `num_particles, dim` | runner (IC generation), ParticleTorch | System size |
| `model_path` | runner (simulation load) | Checkpoint location |
| `save_name` | runner (checkpoint save) | Output directory prefix |

---

## Data Format Conventions

### Initial Conditions
**Format**: (N, 1 + 2*D) numpy array
- Column 0: mass
- Columns 1 to D: position components (x, y, ...)
- Columns D+1 to 2*D: velocity components (vx, vy, ...)

Example 2D binary:
```python
# (2, 5): 2 particles, 2D
[[m1, x1, y1, vx1, vy1],
 [m2, x2, y2, vx2, vy2]]
```

### Features
**Single system**: 1D tensor of length F (feature size)
**Batched**: (B, F) for B systems
**With history**: (K+1, F) concatenated for K+1 time steps

### Model Output
**Shape**: (B, 2) or (1, 2) with 2 components
- Component 0: dt (time step, positive after Softplus)
- Component 1: Auxiliary parameter (often unused or for energy prediction)

---

## Environment & Paths

- **Project root**: `/u/gkerex/projects/AITimeStepper/`
- **Data symlink**: `data/ → /work/hdd/bfpt/gkerex/AITimeStepper/`
- **W&B logs**: `run/wandb/`, `wandb/` (local and offline)
- **Notebooks**: `jupyter_notebooks/`
- **SLURM logs**: Depend on cluster setup (typically in run output)

---

## Dependency Flow

```
runner.py (entry point)
├─ Config (src/config.py)
├─ ModelAdapter (src/model_adapter.py)
│  ├─ ParticleTorch (src/particle.py)
│  ├─ HistoryBuffer (src/history_buffer.py)
│  └─ nbody_features (src/nbody_features.py)
├─ FullyConnectedNN (src/structures.py)
├─ loss_fn_batch* (src/losses.py, src/losses_history.py)
│  └─ [uses ParticleTorch, HistoryBuffer, nbody_features]
├─ save_checkpoint (src/checkpoint.py)
└─ simulators/nbody_simulator.py (inference loop)
   ├─ Particle (simulators/particle.py)
   ├─ evolve_particles_ml*
   └─ [uses ModelAdapter, FullyConnectedNN]
```

---

## Extension Checklist

When adding new functionality:

1. **New feature type**: Add enum value to Config, implement in nbody_features.py, register in HistoryBuffer
2. **New loss function**: Create in losses.py/losses_history.py, call from runner.py
3. **New external field**: Subclass ExternalField, initialize in runner.py
4. **New model architecture**: Subclass nn.Module in structures.py, maintain 2-output contract
5. **New integrator**: Add method to ParticleTorch or function in simulators/, call from runner
6. **New observable**: Add computation function, track in runner's metric lists
7. **Configuration option**: Add field to Config dataclass, add CLI arg in add_cli_args()
