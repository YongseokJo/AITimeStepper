# AITimeStepper Testing Patterns

## Current State: No Formal Test Suite

**As of the latest commits, this project has:**
- No pytest, unittest, or other test framework installed
- No dedicated `tests/` directory
- No CI/CD pipeline (github actions, etc.)
- Test-like patterns are confined to legacy scripts and ad-hoc validation

**Test coverage tools configured but unused:**
- `.gitignore` includes pytest cache, coverage configs, nosetests
- Setup suggests testing *was* planned but not implemented

## Testing Approach

### Manual Validation Scripts (Legacy)

**Legacy scripts in `run/legacy/`:**
- `integration_sanity.py` - basic integration sanity check
- `tidal_sanity.py` - verifies tidal field implementation
- `ML_history_wandb.py` - historical training script with logging
- `ML_history_multi_wandb.py` - multi-orbit training variant

**Status:** Deleted from main branch; kept for reference only

### Command-Line Testing

**Quick sanity check via CLI:**
```bash
python run/runner.py simulate --num-particles 3 --steps 200
```
Minimal integration test; verifies basic simulator works.

**Training quick test:**
```bash
python run/runner.py train --epochs 10 --n-steps 2 --num-particles 4
```
Tests training loop with minimal data; no assertions.

**SLURM job testing:**
```bash
sbatch run/run_ml.slurm      # GPU training
sbatch run/run_sim.slurm     # CPU simulation
```
Tests cluster integration; outputs to log files for manual inspection.

### Physics Validation Approach

**Energy/Momentum Conservation:**
- Computed after each simulation step
- Printed to JSON stdout:
  ```python
  print(json.dumps({
      "energy_initial": energy_initial,
      "energy_final": energy_final,
      "energy_residual": energy_residual,
      "momentum_residual": momentum_residual,
      "angular_momentum_residual": angular_residual,
  }, indent=2))
  ```
- Manually inspected for reasonableness (typically `< 1e-3` for good integrators)

**Model output sanity:**
- Predicted dt > 0 (enforced by Softplus activation)
- Energy/momentum change stays within loss bounds
- No NaN/inf propagation through integrator

## Testing Architecture Patterns

### 1. Configuration Validation

**Pattern: Explicit validate() calls**
```python
def validate(self) -> None:
    if self.history_len and self.history_len > 0 and not self.feature_type:
        raise ValueError("history_len > 0 requires feature_type")
    if self.num_particles is not None and self.num_particles < 2:
        raise ValueError("num_particles must be >= 2")
    if self.dim is not None and self.dim < 1:
        raise ValueError("dim must be >= 1")
```

**Called early in all entry points:**
```python
def run_training(config: Config) -> None:
    if config.integrator_mode == "history" and (config.history_len is None or config.history_len <= 0):
        raise ValueError("history integrator_mode requires history_len > 0")
    config.validate()
```

**How to test:**
```python
# Would create config and call validate() - not yet formalized
config = Config(num_particles=1)  # Invalid
config.validate()  # Raises ValueError
```

### 2. Checkpoint I/O Testing

**Pattern: Load after save**
```python
# Checkpoint structure validated implicitly
def load_model_state(model: torch.nn.Module, path: str | Path, **kwargs) -> Dict[str, Any]:
    ckpt = load_checkpoint(path, map_location=map_location)
    state = ckpt.get("model_state_dict") or ckpt.get("model_state")
    if state is None:
        raise KeyError("Checkpoint missing model_state_dict/model_state")
    model.load_state_dict(state, strict=strict)
    return ckpt
```

**Auto-recovery from checkpoint metadata:**
```python
if config.model_path:
    ckpt_config = load_config_from_checkpoint(config.model_path)
    if ckpt_config is not None:
        if config.history_len == 0 and ckpt_config.history_len:
            config.history_len = ckpt_config.history_len
        if config.feature_type == Config.feature_type and ckpt_config.feature_type:
            config.feature_type = ckpt_config.feature_type
```

**How to test:**
```python
# Manual: Save checkpoint, load with different config, verify recovery
config1 = Config(history_len=5, feature_type="delta_mag")
save_checkpoint("test.pt", model, config=config1)

config2 = Config()
config2.model_path = "test.pt"
loaded_config = load_config_from_checkpoint("test.pt")
assert loaded_config.history_len == 5
```

### 3. Tensor Shape Validation

**Pattern: Explicit shape checks at API boundaries**
```python
def system_features(..., position: torch.Tensor, velocity: torch.Tensor, ...):
    if pos.dim() < 2:
        raise ValueError(f"position must have shape (..., N, D), got {pos.shape}")
    if vel.shape != pos.shape:
        raise ValueError(f"velocity shape {vel.shape} must match position shape {pos.shape}")
```

**PointMassTidalField validation:**
```python
def acceleration(self, position: torch.Tensor, time: Union[float, torch.Tensor] = 0.0) -> torch.Tensor:
    if position.dim() < 2:
        raise ValueError(f"position must have shape (..., N, D), got {tuple(position.shape)}")
    R = self._R(position, time)
    if R.dim() != 1:
        raise ValueError(f"R must be 1D (D,), got {tuple(R.shape)}")
```

**How to test:**
```python
# Would need to pass invalid tensor shapes
pos = torch.randn(5)  # 1D, invalid
vel = torch.randn(5, 2)
system_features(pos, vel, mass=1.0)  # Raises ValueError
```

### 4. Physics Correctness

**Pattern: Inline assertions and diagnostic prints**
```python
# In loss_fn_batch
p = particle.clone_detached()
batch = p._get_batch_()
if batch.dim() == 1:
    batch = batch.unsqueeze(0)

params = model(batch)
if params.dim() == 1:
    params = params.unsqueeze(0)

# Assume shape (B, 2)
dt_raw = params[:, 0]
E0 = p.total_energy_batch(G=1.0)
if E0.dim() == 0:
    E0 = E0.unsqueeze(0)

# Evolve and compute loss
for _ in range(n_steps):
    p.acceleration = p.get_acceleration(G=1.0)
    p.evolve_batch(G=1.0)

E1 = p.total_energy_batch(G=1.0)
```

**Diagnostics printed (but not asserted):**
```python
print("E0: ", E0, "E1: ", E1, "dt: ", dt, "E: ", E)
print("rel_dE_safe:", torch.log(rel_dE_safe).item(), "target_log_rel:", target_log_rel)
```

**How to test:**
```python
# Would verify energy change is small for conservative integrator
E_change = (E1 - E0).abs() / E0.abs()
assert E_change < 1e-2  # Typical tolerance
```

### 5. Model Feature Extraction

**Pattern: Input dimension derivation**
```python
def input_dim_from_state(self, state: ParticleTorch, **kwargs) -> int:
    with torch.no_grad():
        feats = self.build_feature_tensor(state, **kwargs)
        if feats.dim() == 1:
            return int(feats.numel())
        return int(feats.shape[-1])
```

**Feature mode selection:**
```python
def feature_mode(self) -> str:
    if self.config.feature_type in ("basic", "rich"):
        return self.config.feature_type
    return "basic"
```

**How to test:**
```python
adapter = ModelAdapter(config, device="cpu", dtype=torch.float64)
state = ParticleTorch.from_tensors(mass=m, position=pos, velocity=vel)
input_dim = adapter.input_dim_from_state(state)
assert input_dim == expected_dim  # e.g., 11 for "basic"
```

### 6. History Buffer Consistency

**Pattern: State normalization**
```python
@staticmethod
def _normalize_mass(mass: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
    target_shape = position.shape[:-1]  # (..., N)
    if mass.dim() == 0:
        return mass.expand(target_shape)
    if mass.shape == target_shape:
        return mass
    if mass.dim() == 1 and mass.shape[0] == target_shape[-1]:
        view_shape = (1,) * (len(target_shape) - 1) + (target_shape[-1],)
        return mass.reshape(view_shape).expand(target_shape)
    # ... more cases ...
    if mass.shape[-1] != target_shape[-1]:
        raise ValueError(f"mass shape {mass.shape} incompatible with position shape {position.shape}")
    return mass
```

**How to test:**
```python
# Test mass normalization with various inputs
mass_scalar = torch.tensor(1.0)
pos = torch.randn(5, 3, 2)
normalized = HistoryBuffer._normalize_mass(mass_scalar, pos)
assert normalized.shape == (5, 3)

mass_1d = torch.randn(5)
normalized = HistoryBuffer._normalize_mass(mass_1d, pos)
assert normalized.shape == (5, 3)
```

### 7. Numerical Stability

**Pattern: Epsilon guards**
```python
eps = 1e-12
E0_safe = E0 + eps * E0.detach().abs() + eps
rel_dE = torch.abs((E1 - E0) / E0_safe)

# Replace inf/nan
rel_dE = torch.where(
    torch.isfinite(rel_dE),
    rel_dE,
    torch.full_like(rel_dE, 1.0)
)
```

**How to test:**
```python
# Test with E0 near zero
E0 = torch.tensor(1e-15)
E1 = torch.tensor(2e-15)
E0_safe = E0 + 1e-12 * E0.abs() + 1e-12
rel_dE = (E1 - E0) / E0_safe
assert not torch.isnan(rel_dE)
assert not torch.isinf(rel_dE)
```

## Recommended Testing Framework

### Framework: pytest

**Rationale:**
- Lightweight, widely-used in PyTorch ecosystem
- Fixtures for device/dtype setup
- Parameterized testing for tensor shapes
- Benchmark plugin for performance tracking

### Suggested Test Structure

```
tests/
├── conftest.py              # Fixtures (device, dtype, configs)
├── test_config.py           # Configuration validation
├── test_checkpoint.py       # I/O and metadata recovery
├── test_particle.py         # ParticleTorch initialization, integration
├── test_features.py         # Feature extraction, shapes
├── test_losses.py           # Loss computation, physics bounds
├── test_model_adapter.py    # Feature construction, input_dim
├── test_history_buffer.py   # State normalization, feature stacking
├── test_nbody_simulator.py  # NumPy simulator, conservation laws
├── test_external_fields.py  # Tidal field validation
└── integration/
    ├── test_train_loop.py   # End-to-end training
    ├── test_sim_loop.py     # End-to-end simulation
    └── test_checkpoint_recovery.py  # Save/load/resume cycle
```

### Example Fixtures (conftest.py)

```python
import pytest
import torch
import numpy as np
from src import Config, ParticleTorch

@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)

@pytest.fixture(params=[torch.float32, torch.float64])
def dtype(request):
    return request.param

@pytest.fixture
def config_default():
    return Config()

@pytest.fixture
def config_history():
    return Config(history_len=3, feature_type="delta_mag")

@pytest.fixture
def simple_particle(device, dtype):
    pos = torch.randn(4, 2, device=device, dtype=dtype)
    vel = torch.randn(4, 2, device=device, dtype=dtype)
    mass = torch.ones(4, device=device, dtype=dtype)
    return ParticleTorch.from_tensors(mass=mass, position=pos, velocity=vel)
```

### Example Test Cases

```python
# tests/test_config.py
def test_config_validation_history_requires_feature_type():
    config = Config(history_len=5, feature_type="")
    with pytest.raises(ValueError, match="history_len > 0 requires feature_type"):
        config.validate()

def test_config_num_particles_validation():
    config = Config(num_particles=1)
    with pytest.raises(ValueError, match="num_particles must be >= 2"):
        config.validate()

def test_config_cli_args(tmp_path):
    parser = argparse.ArgumentParser()
    Config.add_cli_args(parser, include=["train"])
    args = parser.parse_args(["--epochs", "100", "--lr", "0.001"])
    config = Config.from_cli(args)
    assert config.epochs == 100
    assert config.lr == 0.001

# tests/test_particle.py
def test_particle_initialization(device, dtype):
    pos = torch.randn(4, 2, device=device, dtype=dtype)
    vel = torch.randn(4, 2, device=device, dtype=dtype)
    mass = torch.ones(4, device=device, dtype=dtype)
    particle = ParticleTorch.from_tensors(mass=mass, position=pos, velocity=vel)

    assert particle.position.shape == (4, 2)
    assert particle.velocity.shape == (4, 2)
    assert particle.acceleration.shape == (4, 2)

def test_particle_evolve_shape(simple_particle):
    pos_before = simple_particle.position.clone()
    simple_particle.dt = torch.tensor(0.01)
    simple_particle.evolve()

    assert simple_particle.position.shape == pos_before.shape
    assert simple_particle.velocity.shape == pos_before.shape

def test_total_energy_conservation(simple_particle):
    E0 = simple_particle.total_energy()
    simple_particle.dt = torch.tensor(0.001)

    for _ in range(10):
        simple_particle.evolve()

    E1 = simple_particle.total_energy()
    relative_change = (E1 - E0).abs() / E0.abs()
    assert relative_change < 0.01  # Expect < 1% change over 10 steps

# tests/test_features.py
@pytest.mark.parametrize("mode", ["basic", "rich"])
def test_system_features(simple_particle, mode):
    from src.nbody_features import system_features

    feats = system_features(
        position=simple_particle.position,
        velocity=simple_particle.velocity,
        mass=simple_particle.mass,
        mode=mode
    )

    expected_dims = {"basic": 11, "rich": 23}
    assert feats.shape[-1] == expected_dims[mode]

def test_system_features_invalid_mode(simple_particle):
    from src.nbody_features import system_features

    with pytest.raises(ValueError, match="Unsupported mode"):
        system_features(
            position=simple_particle.position,
            velocity=simple_particle.velocity,
            mass=simple_particle.mass,
            mode="invalid"
        )

# tests/test_checkpoint.py
def test_checkpoint_save_load(tmp_path, simple_particle):
    from src.structures import FullyConnectedNN
    from src.checkpoint import save_checkpoint, load_checkpoint

    model = FullyConnectedNN(input_dim=11, output_dim=2)
    optimizer = torch.optim.Adam(model.parameters())

    path = tmp_path / "test_ckpt.pt"
    save_checkpoint(
        path, model, optimizer,
        epoch=5, loss=torch.tensor(0.123),
        config=Config(history_len=3)
    )

    ckpt = load_checkpoint(path)
    assert ckpt["epoch"] == 5
    assert abs(ckpt["loss"] - 0.123) < 1e-6
    assert ckpt["history_len"] == 3

# tests/test_model_adapter.py
def test_model_adapter_input_dim(simple_particle):
    from src.model_adapter import ModelAdapter

    config = Config(history_len=0, feature_type="basic")
    adapter = ModelAdapter(config, device="cpu", dtype=torch.float64)

    input_dim = adapter.input_dim_from_state(simple_particle)
    assert input_dim == 11  # "basic" mode

def test_model_adapter_history(simple_particle):
    from src.model_adapter import ModelAdapter
    from src.history_buffer import HistoryBuffer

    config = Config(history_len=3, feature_type="delta_mag")
    adapter = ModelAdapter(config, device="cpu", dtype=torch.float64)

    # Push history
    for _ in range(3):
        adapter.history_buffer.push(simple_particle)

    input_dim = adapter.input_dim_from_state(simple_particle)
    assert input_dim > 0
```

### Integration Test Example

```python
# tests/integration/test_train_loop.py
def test_training_one_epoch(simple_particle):
    from src.structures import FullyConnectedNN
    from src.losses import loss_fn_batch

    config = Config(epochs=1, n_steps=2, lr=1e-3)

    model = FullyConnectedNN(
        input_dim=11, output_dim=2,
        hidden_dims=[64, 64]
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    loss_before = None
    for epoch in range(1):
        loss, metrics = loss_fn_batch(model, simple_particle, n_steps=config.n_steps)
        if loss_before is None:
            loss_before = loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert loss_before is not None
    assert "rel_dE" in metrics
    assert metrics["dt"] > 0  # Softplus ensures positivity
```

## Coverage Goals

**Target coverage (if implemented):**
- Core physics modules (particle, features, losses): **90%+**
- Configuration & I/O: **95%+**
- Model components: **85%+**
- Simulators: **80%+**
- CLI/integration: **50%+** (manual testing sufficient)

**Known limitations:**
- GPU-specific tests require CUDA availability
- External field testing limited by mock complexity
- Multi-orbit training requires large memory; only smoke tests

## Performance Benchmarking

**Optional: pytest-benchmark plugin**
```python
def test_feature_extraction_performance(benchmark, simple_particle):
    from src.nbody_features import system_features

    def extract():
        return system_features(
            position=simple_particle.position,
            velocity=simple_particle.velocity,
            mass=simple_particle.mass,
        )

    result = benchmark(extract)
    assert result.shape[-1] == 11
```

## Continuous Integration (Future)

**Recommended GitHub Actions workflow:**
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11"]
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install torch pytest pytest-cov pytest-xdist
      - run: pytest tests/ -v --cov=src --cov-report=xml
      - uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
```

## Summary

**Current state:**
- No formal testing framework
- Manual validation via CLI and print statements
- Physics properties (energy, momentum) monitored informally

**Recommended next steps:**
1. Set up pytest with fixtures for device/dtype
2. Write validation tests for Config, checkpoint I/O, tensor shapes
3. Add physics correctness tests (energy conservation)
4. Integrate into GitHub Actions for CI/CD
5. Aim for 80%+ coverage of core modules

**Key testing philosophy:**
- Physics-first: verify conservation laws hold
- Shape-first: catch dimensional mismatches early
- Configuration-first: validate inputs before computation
- Checkpoint-centric: ensure model persistence works correctly
