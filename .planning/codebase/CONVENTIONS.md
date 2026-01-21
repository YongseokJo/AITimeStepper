# AITimeStepper Code Conventions

## Code Style & Organization

### Python Version & Imports
- Uses `from __future__ import annotations` for forward reference type hints (modern Python 3.7+ style)
- All imports organized in standard order: future, stdlib, third-party, local
- Local imports use relative paths: `from .module import Class`
- Modules grouped by category: config, physics/math, ML, utilities

### File Organization
```
src/               # Core differentiable PyTorch modules
├── config.py      # Centralized @dataclass configuration
├── particle.py    # ParticleTorch - batched N-body state
├── nbody_features.py  # System-level feature extraction
├── history_buffer.py  # HistoryBuffer - temporal state tracking
├── losses.py      # Physics-informed loss functions
├── losses_history.py  # History-aware loss variants
├── model_adapter.py  # Feature construction abstraction
├── structures.py  # Neural network architectures
├── checkpoint.py  # Save/load utilities
├── external_potentials.py  # Tidal fields, external forces
└── __init__.py    # Wildcard re-exports (*)

simulators/        # NumPy-based inference simulators
├── particle.py    # SimParticle - numpy state
└── nbody_simulator.py  # Integration utilities

run/               # Entry points
├── runner.py      # Unified train/simulate CLI
└── run_*.slurm    # Cluster job scripts
```

### Naming Conventions

#### Variables
- **Position/Velocity**: `pos`, `vel`, `position`, `velocity` (full names in signatures)
- **Tensors**: `x`, `v`, `a` (short) or `position`, `velocity`, `acceleration` (full)
- **Batch dimensions**: `B` for batch size, `N` for particles, `D` for spatial dims
- **Time step**: `dt` (scalar or batch)
- **Gravitational constant**: `G` (often hardcoded as 1.0)

#### Classes
- **Physics objects**: `ParticleTorch`, `HistoryBuffer`, `PointMassTidalField`
- **Models**: `FullyConnectedNN`, `SimpleNN`
- **Utilities**: `ModelAdapter`, `Config`
- **PascalCase for all class names**

#### Functions
- **Lower snake_case**: `compute_acceleration()`, `system_features()`, `loss_fn_batch()`
- **Private/internal**: `_normalize_mass()`, `_state_from_tensors()` (leading underscore)
- **Predicates**: none explicit, but validation functions raise exceptions instead

#### Constants
- **UPPER_SNAKE_CASE** rarely used; mostly class fields with lowercase defaults
- **Physics constants**: `G=1.0`, `softening=0.0`, `eps=1e-12` as function parameters
- **Magic numbers**: stored as defaults in Config dataclass

### Type Hints

**Modern style throughout:**
```python
def function(x: torch.Tensor, y: int | str, optional_val: Optional[float] = None) -> Dict[str, Any]:
    ...
```

**Key patterns:**
- Uses `|` for union types (PEP 604) instead of `Union[X, Y]`
- Optional parameters: `Optional[Type] = None`
- Complex returns: `Dict[str, Any]`, `Tuple[torch.Tensor, ...]`
- TYPE_CHECKING blocks for forward references and cycle prevention
- Protocol classes for abstract interfaces (e.g., `ExternalField`)

**Dataclass typing:**
```python
@dataclass
class Config:
    epochs: int = 1000
    history_len: int = 0
    feature_type: str = "delta_mag"
    external_field_position: Optional[Tuple[float, float, float]] = None
```

### Documentation

**Docstrings: Google-style with comments for complex logic**
```python
def loss_fn_batch(
    model,
    particle: "ParticleTorch",
    n_steps: int = 1,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    One forward pass + physics loss.

    model: maps batch -> [dt_raw, E_hat_raw]
    particle: ParticleTorch object (single or batched)
    n_steps: integration steps for predicted dt

    Returns:
        loss: scalar tensor
        metrics: dict of diagnostics
    """
```

**Inline comments for physics/math:**
- Clearly explain tensor shapes: `# (..., N, N, D)`, `# (B, N)`
- Explain numerical tricks: `# Zero self-interaction by sending diagonal to infinity`
- Mark workarounds/hacks: `# Workaround for...` or `# TODO: optimize...`

### Tensor Shapes

**Consistent notation in docstrings and comments:**
- `(N, D)` - N particles, D spatial dimensions
- `(B, N, D)` - batch of B systems
- `(..., N, D)` - arbitrary leading dimensions (e.g., time steps, ensembles)
- Scalar tensors: `()` or noted as "scalar tensor"

**Shape validation:**
```python
if pos.dim() < 2:
    raise ValueError(f"position must have shape (..., N, D), got {pos.shape}")
if vel.shape != pos.shape:
    raise ValueError(f"velocity shape {vel.shape} must match position shape {pos.shape}")
```

## Error Handling & Validation

### Validation Strategy

**Config validation via `Config.validate()` method:**
```python
def validate(self) -> None:
    if self.history_len and self.history_len > 0 and not self.feature_type:
        raise ValueError("history_len > 0 requires feature_type")
    if self.num_particles is not None and self.num_particles < 2:
        raise ValueError("num_particles must be >= 2")
    if self.dim is not None and self.dim < 1:
        raise ValueError("dim must be >= 1")
    if self.duration is not None and self.duration < 0:
        raise ValueError("duration must be >= 0")
```

**Early validation in top-level functions:**
```python
def run_training(config: Config) -> None:
    if config.integrator_mode == "history" and (config.history_len is None or config.history_len <= 0):
        raise ValueError("history integrator_mode requires history_len > 0")
    config.validate()
```

### Exception Types

- **ValueError**: Invalid parameters, shape mismatches, unsupported modes
- **KeyError**: Missing checkpoint data (model_state_dict, config)
- **RuntimeError**: Import failures (e.g., wandb not installed)
- **No custom exceptions** - uses standard Python exceptions consistently

### Numerical Safety

**Epsilon handling:**
```python
eps = 1e-12  # Prevent division by zero
E0_safe = E0 + eps * E0.detach().abs() + eps
rel_dE = (E1 - E0) / E0_safe
```

**Inf/NaN replacement:**
```python
rel_dE = torch.where(
    torch.isfinite(rel_dE),
    rel_dE,
    torch.full_like(rel_dE, 1.0)  # Large penalty
)
```

**Softening for gravitational singularities:**
- Applied as: `dist2 = dist2 + softening**2`
- Prevents numerical blow-up in close encounters

## PyTorch Patterns

### Tensor Operations

**Out-of-place operations preferred:**
- Enables autograd graph construction: `self.position = x_new` (not in-place `+=`)
- Comments note when operations are in-place: `# Update self.position (out-of-place)`

**Broadcasting conventions:**
```python
# Unsqueeze for broadcasting
pos_i = pos.unsqueeze(-2)  # (..., N, 1, D)
pos_j = pos.unsqueeze(-3)  # (..., 1, N, D)
r_ij = pos_j - pos_i       # (..., N, N, D)
```

**Detach & clone for gradient isolation:**
```python
p = particle.clone_detached()  # Fresh copy, no grad
dt_for_time = dt_for_time.detach()  # Scalar bookkeeping only
```

### Device & Dtype Handling

**Config resolution methods:**
```python
def resolve_device(self) -> torch.device:
    if self.device == "cpu":
        return torch.device("cpu")
    if self.device == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def resolve_dtype(self) -> torch.dtype:
    if str(self.dtype) == "float32":
        return torch.float32
    return torch.float64
```

**Consistent tensor creation:**
```python
tensor = torch.as_tensor(value, dtype=dtype, device=device)
# Not: torch.tensor(value)  (uses default device/dtype)
```

**torch.no_grad() blocks for inference:**
```python
model.eval()
with torch.no_grad():
    output = model(input)
```

### Model Architecture

**FullyConnectedNN pattern:**
```python
class FullyConnectedNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[64, 64],
                 activation='relu', dropout=0.0, output_positive=True):
        # Builds sequential layers dynamically
        layers = []
        for h in hidden_dims:
            layers.append(nn.Linear(current_dim, h))
            layers.append(act_fn())  # Activation
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(current_dim, output_dim))
        if output_positive:
            layers.append(nn.Softplus())  # Force positive outputs
        self.model = nn.Sequential(*layers)
```

## Configuration Management

### Config as Dataclass

**Central Config class in `src/config.py`:**
- All hyperparameters, paths, and settings as class fields
- Defaults provided; no magic numbers scattered through code
- Methods for conversion: `from_dict()`, `to_dict()`, `from_cli()`, `as_wandb_dict()`

**Sections organized by comment:**
```python
@dataclass
class Config:
    # Training / optimization
    optimizer: str = "adam"
    epochs: int = 1000
    lr: float = 1e-4

    # Loss bounds
    E_lower: float = 1e-6
    E_upper: float = 1e-4

    # Feature / history
    history_len: int = 0
    feature_type: str = "delta_mag"

    # Simulation / integrator
    integrator_mode: str = "analytic"
    dt: float = -1.0

    # Device / dtype
    device: str = "auto"
    dtype: str = "float64"

    # Extra / unknown fields
    extra: Dict[str, Any] = field(default_factory=dict)
```

**CLI argument mapping:**
```python
@classmethod
def add_cli_args(cls, parser: argparse.ArgumentParser,
                 include: Optional[Iterable[str]] = None) -> None:
    # Filters args by category (train, bounds, history, etc.)
    if want("train"):
        add_arg("--epochs", "-n", type=int, default=cls.epochs, ...)
```

### Checkpoint Format

**Unified checkpoint structure:**
```python
payload = {
    "epoch": epoch,
    "model_state_dict": model_state,
    "optimizer_state_dict": optimizer_state,
    "loss": scalar,
    "logs": metrics_dict,
    "config": config.as_wandb_dict(),
    "history_len": config.history_len,
    "feature_type": config.feature_type,
    "dtype": config.dtype,
}
torch.save(payload, path)
```

**Auto-loading from checkpoint:**
```python
if config.model_path:
    ckpt_config = load_config_from_checkpoint(config.model_path)
    if ckpt_config is not None:
        if config.history_len == 0 and ckpt_config.history_len:
            config.history_len = ckpt_config.history_len
```

## Logging & Monitoring

### Print-based Debugging
- Uses `print()` for diagnostic output (no logging module)
- Verbose output in loss functions: `print("params: ", params, "params shape: ", params.shape)`
- JSON output for structured simulation results (stdout)

### Weights & Biases Integration (Optional)
```python
if config.extra.get("wandb", False):
    import wandb as wandb_lib
    wandb_run = wandb.init(project=wandb_project, name=wandb_name, config=config.as_wandb_dict())

    # Log metrics per epoch
    wandb.log({"epoch": epoch, "loss": float(loss.item()), "dt": dt, "rel_dE": rel_dE})
```

### Metrics & Results
**Per-epoch logs dict:**
```python
metrics = {
    "rel_dE": rel_dE.mean(),
    "dt": dt.mean(),
    "E0": E0.mean(),
    "loss_energy": loss_energy.mean(),
    "rel_dE_mean": rel_dE_mean.mean(),
    "loss_pred": loss_pred.mean(),
}
```

**Simulation results as JSON:**
```python
print(json.dumps({
    "steps": step_count,
    "energy_initial": energy_initial,
    "energy_residual": energy_residual,
    "momentum_residual": momentum_residual,
    "angular_momentum_residual": angular_residual,
}, indent=2))
```

## Physics & Math Conventions

### Gravitational Physics
- **G = 1.0**: Hard-coded in most places; parameter passed as `G=1.0`
- **Softening**: Added to prevent singularities: `dist2 = dist2 + softening**2`
- **Units**: Simulation units (G=1, masses in units of M☉, distances arbitrary)

### Integration Methods
- **Leapfrog/Velocity-Verlet**: symplectic integrator for energy conservation
- **Steps**: n_steps integration steps per loss evaluation during training
- **Adaptive dt**: Model predicts dt; system evolves with this step size

### Conservation Laws Monitored
- **Energy**: Total E = KE + PE; tracked as `(E1 - E0) / E0` (relative change)
- **Momentum**: Total p = sum(m*v); tracked as norm of momentum
- **Angular momentum**: L = sum(r × (m*v)); tracked for 2D/3D systems

### Loss Bounds (Band Loss)
```python
def band_loss_zero_inside_where(rel_dE, E_lower, E_upper):
    loss_below = (E_lower - rel_dE).clamp(min=0)**2
    loss_above = (rel_dE - E_upper).clamp(min=0)**2
    return loss_below + loss_above
```
Penalizes energy change outside tolerance band; zero inside.

## Common Patterns

### Particle State Management

**Two constructors for ParticleTorch:**
```python
# For inference/quick tests
particle = ParticleTorch(mass=m, position=pos, velocity=vel)

# For training (preserves gradients)
particle = ParticleTorch.from_tensors(mass=m, position=pos, velocity=vel)
```

**State cloning for loss computation:**
```python
p = particle.clone_detached()  # Fresh copy, no graph
# ... modify p during integration ...
# particle remains unchanged
```

### Batch Handling

**Single vs. Batch logic:**
```python
if batch.dim() == 1:
    batch = batch.unsqueeze(0)  # (F,) -> (1, F)

params = model(batch)  # (B, 2)

if params.dim() == 1:
    params = params.unsqueeze(0)  # (2,) -> (1, 2)
```

**Feature extraction modes:**
- `"basic"`: N, D, total_mass, r_mean, r_max, v_mean, v_max, a_mean, a_max, pair_min, pair_mean (11 features)
- `"rich"`: richer stats including min/max/rms per quantity (23 features)
- `"delta_mag"`: time-concatenated changes in position/velocity/acceleration

### History Buffer Integration

**Temporal feature stacking:**
```python
history_buffer = HistoryBuffer(history_len=5, feature_type="delta_mag")
history_buffer.push(particle)
features = history_buffer.features_for(particle)  # (K+1)*F features
```

## Summary

**Key takeaways:**
1. **Dataclass-driven configuration** - all parameters centralized, validated, serializable
2. **Out-of-place tensor ops** - enables autograd through compute graphs
3. **Shape consistency** - $(B, N, D)$ notation pervasive; validated at boundaries
4. **Physics-informed** - losses designed around conservation laws
5. **Minimal logging** - print() and JSON; optional W&B integration
6. **Type hints** - modern syntax (PEP 604), comprehensive coverage
7. **Error handling** - early validation, clear ValueError/KeyError messages
8. **Numerical stability** - epsilon guards, softening, inf/nan replacement
