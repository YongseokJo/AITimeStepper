# ARCHITECTURE.md - AITimeStepper System Design

## Project Overview

AITimeStepper is a physics-informed ML time stepper for N-body gravitational simulations. It trains neural networks to predict adaptive time steps (dt) for efficient N-body integration while preserving energy and momentum conservation.

## Core Architecture Layers

### Layer 1: Configuration & State Management

**Module**: `src/config.py` - `Config` dataclass

The centralized configuration system manages all parameters across training, simulation, and device settings:

- **Training parameters**: epochs, learning rate, optimization strategy, loss bounds (E_lower/E_upper, L_lower/L_upper)
- **Feature configuration**: history length, feature types (basic/rich/delta_mag)
- **Physics parameters**: dt bounds, number of integration steps per forward pass, loss thresholds
- **System parameters**: particle count, dimensionality, masses, softening
- **External fields**: optional tidal field configuration
- **Device settings**: auto device detection, dtype (float32/float64), TF32/compile flags

Key methods:
- `from_dict()` / `to_dict()`: Serialization for checkpointing
- `resolve_device()` / `resolve_dtype()`: Device resolution with auto-detection
- `validate()`: Parameter consistency validation
- `add_cli_args()`: Flexible CLI generation with filtering

**Contract**: Config is saved with checkpoints and auto-loaded during inference to ensure reproducibility.

---

### Layer 2: Differentiable Physics Engine

**Module**: `src/particle.py` - `ParticleTorch` class

Provides batched, autograd-compatible particle state with differentiable N-body dynamics:

**State representation**:
- `position`: shape (N, D) or (B, N, D) - particle positions
- `velocity`: shape (N, D) or (B, N, D) - particle velocities
- `acceleration`: shape (N, D) or (B, N, D) - computed accelerations
- `mass`: scalar or shape (N,) or (B, N) - per-particle masses
- `dt`: time step (scalar or per-batch)
- `current_time`: tracking variable for simulation time

**Key methods**:

- `get_acceleration(G=1.0)`: Computes softened Newtonian gravitational acceleration with external field support. Supports batched operations via pairwise distance tensors.

- `evolve()`: Single leapfrog integration step (symplectic, velocity-Verlet):
  ```
  v_{n+1/2} = v_n + 0.5*a(x_n)*dt
  x_{n+1} = x_n + v_{n+1/2}*dt
  a_{n+1} = a(x_{n+1})
  v_{n+1} = v_{n+1/2} + 0.5*a_{n+1}*dt
  ```

- `total_energy()`: Differentiable kinetic + potential energy computation
- `total_energy_batch()`: Batched energy for multiple independent orbits
- `clone_detached()`: Creates fresh copy with detached graph for new forward pass
- `reset_state()`: In-place state reset for reusing particle object

**Constructors**:
- `__init__()`: For non-training use
- `from_tensors()`: For training - preserves autograd through tensors

**External fields**: `set_external_field()` supports analytical external potentials (e.g., tidal fields) via `ExternalField` abstraction.

---

### Layer 3: Feature Extraction

**Module**: `src/nbody_features.py` - System-level features

Computes fixed-size, permutation-invariant feature vectors for N-body systems:

- **basic mode**: center of mass, inter-particle distances, statistics (mean/std)
- **rich mode**: extended feature set with higher-order statistics
- **delta_mag mode** (temporal): differences in position, velocity, acceleration magnitudes + dt history

Features are extracted via `ParticleTorch.system_features(mode=...)` or `HistoryBuffer` for temporal features.

**Module**: `src/history_buffer.py` - `HistoryBuffer` class

Stores K past particle states (detached tensors) and provides time-concatenated features:

- Maintains deque of `_HistoryState` objects (position, velocity, mass, dt, softening)
- Supports feature types: basic, rich, delta_mag
- `push(state)`: Add detached copy of current state
- `features_for(state)`: Build feature vector from history + current state
- `features_for_histories()`: Multi-orbit batched feature computation

For single-system usage, returns 1D vector of length (K+1) × F, where K is history_len and F is per-step feature size. Pads with oldest state if fewer than K past states available.

---

### Layer 4: Model Adaptation Layer

**Module**: `src/model_adapter.py` - `ModelAdapter` class

Abstracts the interface between differentiable physics and ML models:

- **Initialization**: Resolves device/dtype, optionally creates `HistoryBuffer` based on config
- **Feature abstraction**: Unifies analytic features (system_features) and history-aware features (HistoryBuffer)
- `build_feature_tensor()`: Construct model input from particle state and optional history
- `input_dim_from_state()`: Determine model input dimension from state
- `predict_dt()`: Run model inference in no_grad mode, returns dt prediction
- `update_history()`: Push new state to history buffer if history is enabled

**Properties**:
- `history_enabled`: Boolean flag from config
- `feature_mode()`: Map config.feature_type to actual feature extraction mode

**Responsibility**: Provides unified abstraction so runner code doesn't know whether features are analytic or history-based.

---

### Layer 5: Neural Network Model

**Module**: `src/structures.py` - `FullyConnectedNN` class

Configurable fully-connected neural network for dt prediction:

- **Architecture**: Input → Hidden layers (with activation) → Output (2 outputs for dt and auxiliary parameter)
- **Features**:
  - Configurable hidden layer dimensions
  - Activation options: relu, tanh, sigmoid, silu
  - Dropout support
  - `output_positive=True`: Applies Softplus to ensure positive dt predictions

**Data flow**:
```
features (1D or batched) → fc1 → activation → ... → fcN → output [dt_raw, aux_raw]
                                                                ↓
                                                    softplus → [dt, aux_positive]
```

---

### Layer 6: Physics-Informed Loss Functions

**Module**: `src/losses.py` - Physics loss functions for analytic features

**Core loss functions**:

- `loss_fn_batch()`: Single-orbit loss computation
  - Clones particle state (detach to break graph)
  - Extracts features
  - Runs model → dt, auxiliary parameter
  - Integrates n_steps with predicted dt
  - Computes energy/momentum conservation penalties
  - Returns scalar loss + diagnostic logs

- Energy conservation penalty: `|ΔE/E| - E_hat|²` where E_hat is predicted energy drift, bounded by E_lower/E_upper
- Angular momentum penalty: `|ΔL/L|²` bounded by L_lower/L_upper

**Module**: `src/losses_history.py` - Physics loss for history-aware models

- `loss_fn_batch_history()`: Single-orbit with temporal features
- `loss_fn_batch_history_batch()`: Multi-orbit batched loss with histories

Features temporal state changes via `delta_mag` feature type.

---

### Layer 7: Checkpoint & Serialization

**Module**: `src/checkpoint.py`

Handles model persistence with metadata:

- `save_checkpoint()`: Saves model state, optimizer state, epoch, loss, logs, and Config
  - Stores history_len, feature_type, dtype for auto-loading during inference
  - Creates parent directories as needed

- `load_model_state()`: Loads model weights from checkpoint
- `load_config_from_checkpoint()`: Extracts Config dict for reproducibility

---

### Layer 8: Simulation Engine (NumPy-based Inference)

**Module**: `simulators/nbody_simulator.py`

NumPy-based N-body integrator for fast inference:

- `generate_random_ic()`: Random initial conditions with configurable masses, scales, seeds
- `compute_accelerations()`: Vectorized N-body acceleration computation
- `evolve_step()`: Single leapfrog integration step
- `evolve_particles()`: Fixed-dt evolution loop
- `evolve_particles_ml()`: ML-predicted dt evolution with model inference

Features:
- Energy/momentum/angular momentum tracking
- External field support
- Softening for numerical stability

**Module**: `simulators/particle.py`

NumPy particle representation for inference (distinct from PyTorch's ParticleTorch).

---

### Layer 9: Runner & Entry Point

**Module**: `run/runner.py`

Main orchestrator for training and simulation:

**Training flow** (`run_training()`):
1. Resolve device/dtype
2. Create `ModelAdapter`
3. Initialize `FullyConnectedNN` model
4. Setup optimizer (Adam)
5. For each epoch:
   - Compute loss via appropriate loss function (analytic or history-based)
   - Backprop & optimizer step
   - Log to W&B if enabled
   - Save checkpoint every 10 epochs

**Simulation flow** (`run_simulation()`):
1. Load ICs (random or from file)
2. Create `ModelAdapter`
3. If ml/history mode:
   - Load model from checkpoint
   - Auto-load history_len, feature_type, dtype from checkpoint config
   - Attach model to simulator particles
4. Evolve particles:
   - Fixed dt (analytic mode)
   - ML-predicted dt (ml/history modes)
5. Track energy/momentum/angular momentum conservation
6. Output conservation metrics as JSON

**Modes**:
- `simulate`: Run N-body simulation
- `train`: Train ML time stepper

---

## Data Flow Diagrams

### Training Data Flow

```
ICs (random or file)
    ↓
ParticleTorch (on GPU)
    ↓
ModelAdapter → build_feature_tensor()
    ↓
FullyConnectedNN → predict (dt_raw, aux_raw)
    ↓
loss_fn_batch() or loss_fn_batch_history()
    ├─ ParticleTorch.evolve() × n_steps [leapfrog]
    ├─ ParticleTorch.total_energy() [energy conservation]
    ├─ ParticleTorch.total_energy_batch() [for multi-orbit]
    ↓
Scalar loss (differentiable)
    ↓
Backward pass through entire graph
    ↓
Optimizer.step()
    ↓
save_checkpoint() every 10 epochs
```

### Inference Data Flow

```
ICs (numpy array)
    ↓
SimParticles (simulators/particle.py)
    ↓
evolve_particles_ml()
    ├─ Extract numpy state
    ├─ Convert to ParticleTorch (CPU/GPU)
    ├─ ModelAdapter.predict_dt() [no_grad]
    ├─ Leapfrog step
    ├─ ModelAdapter.update_history() [if history mode]
    ↓
Energy/momentum conservation metrics
    ↓
JSON output
```

### Multi-Orbit Training (Batched)

```
num_orbits=8 random ICs
    ↓
ParticleTorch batch: position (8, N, D), velocity (8, N, D), mass (8, N)
    ↓
stack_particles() → batched ParticleTorch
    ↓
ModelAdapter.build_feature_tensor(batch_state, histories=[...])
    ↓
HistoryBuffer.features_for_histories() [parallel feature extraction]
    ↓
Model input (batched features)
    ↓
loss_fn_batch_history_batch() [multi-orbit loss]
```

---

## Key Design Patterns

### 1. Layered Abstraction
Each layer has single responsibility: physics (particle), features (adapter), model (structures), loss (losses), I/O (checkpoint).

### 2. Unified Interface via ModelAdapter
Analytic and history-aware modes share same interface—runner doesn't care which is active.

### 3. Batching for Efficiency
- Single-orbit training: ParticleTorch (N, D)
- Multi-orbit training: ParticleTorch (B, N, D) with histories
- Parallel feature extraction via HistoryBuffer.features_for_histories()

### 4. Autograd Integration
- Training: Graph flows through particle evolution → loss → backward
- Inference: Model runs in no_grad, particle evolution detached

### 5. Checkpoint Contract
Config saved with model → inference auto-configures history_len, feature_type, dtype from checkpoint.

### 6. External Field Support
`ParticleTorch.external_field` allows injecting analytical potentials (tidal fields) without modifying physics engine.

---

## Dependency Graph

```
runner.py (entry point)
    ├─ config.py (Config class)
    ├─ model_adapter.py (ModelAdapter)
    │   ├─ particle.py (ParticleTorch)
    │   ├─ history_buffer.py (HistoryBuffer)
    │   ├─ nbody_features.py (feature extraction)
    │   └─ config.py (Config)
    ├─ structures.py (FullyConnectedNN)
    ├─ losses.py or losses_history.py (loss functions)
    │   ├─ particle.py (ParticleTorch methods)
    │   ├─ history_buffer.py (HistoryBuffer)
    │   └─ nbody_features.py (features)
    ├─ checkpoint.py (save/load)
    └─ simulators/
        ├─ nbody_simulator.py (evolve_particles_ml)
        ├─ particle.py (SimParticle)
        └─ (external_acceleration support)
```

---

## Abstraction Boundaries

| Layer | Input | Output | Responsibility |
|-------|-------|--------|-----------------|
| **Config** | CLI args / dict | Config object | Parameter management, validation |
| **ParticleTorch** | pos, vel, mass, dt | pos, vel, accel (evolved) | Differentiable physics |
| **Features** | ParticleTorch + history | Feature tensor | Fixed-size representation |
| **ModelAdapter** | ParticleTorch | Feature tensor, dt | Abstraction over feature types |
| **Model** | Feature tensor | [dt, aux] | Neural network |
| **Loss** | ParticleTorch, model | Scalar loss | Physics-informed training signal |
| **Checkpoint** | Model, Config | File | Persistence |
| **Simulator** | ICs, model, dt | Energy/momentum residuals | Fast inference loop |

---

## Extension Points

1. **New feature types**: Add to `nbody_features.py`, register in Config.feature_type choices
2. **New loss functions**: Add to `losses.py` or `losses_history.py`, call from runner
3. **New external fields**: Subclass `ExternalField`, attach via `particle.set_external_field()`
4. **New models**: Replace FullyConnectedNN with custom `nn.Module`, maintain interface
5. **New integrators**: Add to `simulators/nbody_simulator.py`, call from runner
6. **Device support**: Modify Config.resolve_device() and test on target device

---

## File Sizes and Responsibilities

| File | Lines | Primary Responsibility |
|------|-------|------------------------|
| `config.py` | 274 | Configuration & CLI |
| `particle.py` | 572 | Differentiable physics engine |
| `model_adapter.py` | 115 | Feature abstraction |
| `structures.py` | 80+ | Neural network model |
| `losses.py` | 200+ | Physics-informed training loss (analytic) |
| `losses_history.py` | 200+ | Physics-informed training loss (temporal) |
| `checkpoint.py` | 100 | Persistence |
| `history_buffer.py` | 200+ | Temporal state tracking |
| `nbody_features.py` | 100+ | Feature extraction |
| `integrator.py` | 21 | Integration utilities (mostly toy examples) |
| `external_potentials.py` | 100+ | External field support (tidal) |
| `runner.py` | 429 | Main entry point & orchestration |
| `nbody_simulator.py` | 300+ | NumPy simulation engine |

---

## Architectural Decisions

### Why PyTorch for Training, NumPy for Inference?
- **Training**: Need autograd for backpropagation through physics
- **Inference**: NumPy is faster for large-scale simulations without grad tracking

### Why ModelAdapter Abstraction?
- Cleanly separates analytic vs. history-aware feature extraction
- Runner code remains agnostic to feature type
- Easy to add new feature modes

### Why Clone+Detach in Loss Functions?
- Prevents accidental gradient leakage between forward passes
- Ensures each iteration starts fresh
- Allows reusing ParticleTorch objects across epochs

### Why Fixed-Size Features + HistoryBuffer?
- Neural networks require fixed input dimensions
- History buffer pads older states to maintain history_len
- Compatible with batching and permutation invariance

### Why Symplectic Leapfrog Integration?
- Preserves phase-space volume (good for Hamiltonian systems)
- Second-order accuracy with first-order cost
- Standard in N-body astronomy codes

---

## Known Limitations & Future Work

1. **Single model architecture**: All experiments use same 4-layer FullyConnectedNN; could benefit from architecture search
2. **Fixed softening**: Could be learned or adaptive
3. **No adaptive time stepping in integration**: Uses dt from model every step; could use error estimation
4. **Limited to 2D/3D**: Scalability to higher dimensions not tested
5. **Batch features only**: No support for heterogeneous particle sets
