# Phase 6: Integration into runner.py - Research

**Researched:** 2026-01-21
**Domain:** Runner refactoring / Python module integration
**Confidence:** HIGH

## Summary

Phase 6 integrates the new two-phase training system (from Phase 5) into the existing `run/runner.py` entry point. The research examined the current `run_training()` implementation in detail, identifying the exact patterns for particle initialization, model creation, W&B setup, and checkpointing that must be preserved.

The integration is straightforward because `run_two_phase_training()` was designed to mirror the existing patterns in `runner.py`. The main work involves:
1. Replacing the training loop in `run_training()` with a call to `run_two_phase_training()`
2. Preserving the existing setup code (device/dtype, seeds, particle initialization)
3. Passing the correct save_dir and wandb_run objects to the new function

**Primary recommendation:** Modify `run_training()` to delegate to `run_two_phase_training()` while keeping all setup code. This is a surgical replacement of the inner loop, not a rewrite.

## Standard Stack

### Existing Functions to Preserve

| Component | Location | Purpose | Keep As-Is |
|-----------|----------|---------|------------|
| `ModelAdapter` | `src/model_adapter.py` | Feature construction | YES |
| `FullyConnectedNN` | `src/structures.py` | Model architecture | YES |
| `make_particle` | `src/particle.py` | Create ParticleTorch from ICs | YES |
| `save_checkpoint` | `src/checkpoint.py` | Checkpoint serialization | YES |
| `load_config_from_checkpoint` | `src/checkpoint.py` | Config recovery | YES |

### New Functions to Use

| Function | Location | Purpose | Replaces |
|----------|----------|---------|----------|
| `run_two_phase_training` | `src/unified_training.py` | N-epoch outer loop | Inner `while epoch < config.epochs` loop |
| `train_epoch_two_phase` | `src/unified_training.py` | Single epoch | `loss_fn_batch*` calls + optimizer.step() |

### Supporting Infrastructure (Unchanged)

| Component | Location | Purpose |
|-----------|----------|---------|
| `Config` | `src/config.py` | All parameters |
| `HistoryBuffer` | `src/history_buffer.py` | Temporal features |
| `generate_random_ic` | `simulators/nbody_simulator.py` | Initial conditions |

## Architecture Patterns

### Current run_training() Structure (to preserve)

```python
def run_training(config: Config) -> None:
    # === SECTION 1: VALIDATION (keep) ===
    if config.integrator_mode == "history" and (config.history_len is None or config.history_len <= 0):
        raise ValueError("history integrator_mode requires history_len > 0")
    config.validate()

    # === SECTION 2: DEVICE/DTYPE SETUP (keep) ===
    device = config.resolve_device()
    dtype = config.resolve_dtype()
    torch.set_default_dtype(dtype)
    config.apply_torch_settings(device)

    # === SECTION 3: SEED (keep) ===
    if config.seed is not None:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

    # === SECTION 4: ADAPTER (keep) ===
    adapter = ModelAdapter(config, device=device, dtype=dtype)

    # === SECTION 5: W&B SETUP (keep) ===
    wandb_run = None
    wandb = None
    if config.extra.get("wandb", False):
        # ... initialize wandb ...
        wandb_run = wandb.init(...)

    # === SECTION 6: PARTICLE INIT (keep, with simplification) ===
    # NOTE: Multi-orbit case has special handling - see below

    # === SECTION 7: MODEL CREATION (keep) ===
    model = FullyConnectedNN(...)

    # === SECTION 8: OPTIMIZER (keep) ===
    optimizer = torch.optim.Adam(...)

    # === SECTION 9: TRAINING LOOP (REPLACE) ===
    # OLD: while epoch < config.epochs: ... loss_fn_batch ... optimizer.step()
    # NEW: run_two_phase_training(...)

    # === SECTION 10: W&B CLEANUP (keep) ===
    if wandb_run is not None:
        wandb.finish()
```

### run_two_phase_training() Signature

```python
def run_two_phase_training(
    model: torch.nn.Module,
    particle: ParticleTorch,
    optimizer: torch.optim.Optimizer,
    config: Config,
    adapter: ModelAdapter,
    history_buffer: Optional[HistoryBuffer] = None,
    save_dir: Optional[Path] = None,
    wandb_run: Optional[Any] = None,
    checkpoint_interval: int = 10,
) -> Dict[str, Any]:
```

### Integration Pattern

```python
def run_training(config: Config) -> None:
    # Sections 1-8: UNCHANGED (device, seeds, adapter, wandb, particle, model, optimizer)
    ...

    # Section 9: REPLACE training loop with:
    save_dir = pathlib.Path(project_root) / "data" / (config.save_name or "run_nbody") / "model"

    result = run_two_phase_training(
        model=model,
        particle=particle,  # Single particle for now; multi-orbit deferred
        optimizer=optimizer,
        config=config,
        adapter=adapter,
        history_buffer=adapter.history_buffer,  # Use adapter's buffer
        save_dir=save_dir,
        wandb_run=wandb_run,
        checkpoint_interval=10,
    )

    # Print summary
    print(f"Training complete: {result['epochs_completed']} epochs, "
          f"convergence rate: {result['convergence_rate']:.1%}")

    # Section 10: UNCHANGED
    if wandb_run is not None:
        wandb.finish()
```

### Multi-Orbit Handling (Current Pattern)

The current `run_training()` has complex multi-orbit handling:

```python
if config.num_orbits > 1:
    particles = []
    histories = []
    for i in range(config.num_orbits):
        ptcls = _build_particle()
        particle = make_particle(ptcls, device=device, dtype=dtype)
        particle.current_time = torch.tensor(0.0, device=device, dtype=dtype)
        particles.append(particle)
        if adapter.history_enabled:
            hb = HistoryBuffer(history_len=config.history_len, feature_type=config.feature_type)
            hb.push(particle.clone_detached())
            histories.append(hb)
    batch_state = stack_particles(particles)
    # Then uses loss_fn_batch_history_batch() for training
```

**Recommendation for Phase 6:** Start with single-orbit integration (num_orbits=1). Multi-orbit batching requires changes to `collect_trajectory()` and `generalize_on_trajectory()` to handle batched particles, which is out of scope for this phase.

**Temporary solution:** Add validation that warns/errors if `num_orbits > 1` and suggests using legacy mode or implementing batched support in a future phase.

### Checkpoint Contract (Critical for INTG-03)

From `src/checkpoint.py`, checkpoints include:

```python
payload = {
    "epoch": epoch,
    "model_state_dict": model_state,
    "model_state": model_state,  # Duplicate for compatibility
    "optimizer_state_dict": optimizer_state,
    "optimizer_state": optimizer_state,
    "loss": _tensor_to_value(loss),
    "info": _map_dict_values(info),
    "logs": _map_dict_values(logs),
    "extra": extra,
}

if config is not None:
    payload["config"] = _config_payload(config)
    payload["config_summary"] = config.summary()
    payload["history_len"] = config.history_len
    payload["feature_type"] = config.feature_type
    payload["dtype"] = config.dtype
```

**Simulation mode loads checkpoints via:**
```python
# In run_simulation():
ckpt_config = load_config_from_checkpoint(config.model_path)
if ckpt_config is not None:
    if config.history_len == 0 and ckpt_config.history_len:
        config.history_len = ckpt_config.history_len
    if config.feature_type == Config.feature_type and ckpt_config.feature_type:
        config.feature_type = ckpt_config.feature_type
    if config.dtype == Config.dtype and ckpt_config.dtype:
        config.dtype = ckpt_config.dtype
```

**run_two_phase_training() already preserves this contract** by passing `config=config` to `save_checkpoint()`.

## Config Fields to Preserve

### Training Parameters (Used by new system)

| Field | Default | Used In | Notes |
|-------|---------|---------|-------|
| `epochs` | 1000 | `run_two_phase_training` outer loop | Unchanged |
| `energy_threshold` | 2e-4 | `collect_trajectory_step`, `generalize_on_trajectory` | New parameter |
| `steps_per_epoch` | 1 | `collect_trajectory` | New parameter |
| `replay_steps` | 1000 | `generalize_on_trajectory` max iterations | New parameter |
| `replay_batch_size` | 512 | `generalize_on_trajectory` minibatch | New parameter |
| `E_lower` | 1e-6 | `compute_single_step_loss` | Unchanged |
| `E_upper` | 1e-4 | `compute_single_step_loss` | Unchanged |
| `L_lower` | 1e-4 | Not used in new system | (angular momentum bounds) |
| `L_upper` | 1e-2 | Not used in new system | (angular momentum bounds) |
| `lr` | 1e-4 | Optimizer creation | Unchanged |
| `weight_decay` | 1e-2 | Optimizer creation | Unchanged |
| `history_len` | 0 | `ModelAdapter`, `HistoryBuffer` | Unchanged |
| `feature_type` | "delta_mag" | `ModelAdapter`, `HistoryBuffer` | Unchanged |

### CLI Arguments (Must remain functional)

From `build_parser()` in runner.py:

```python
train = sub.add_parser("train", help="Train ML time-stepper for N-body")
Config.add_cli_args(train, include=["train", "bounds", "history", "device", "logging", "sim", "multi"])
train.add_argument("--ic-path", type=str, default=None, help="path to ICs")
train.add_argument("--wandb", action="store_true", help="enable W&B logging")
train.add_argument("--wandb-project", type=str, default="AITimeStepper")
train.add_argument("--wandb-name", type=str, default=None)
```

All CLI arguments flow through `Config.from_dict(vars(args))` - no changes needed.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Epoch loop with checkpointing | Custom loop in runner.py | `run_two_phase_training()` | Already handles checkpoints, logging, progress |
| W&B metric names | New metric schema | Existing `part1/*`, `part2/*` prefixes | Maintains dashboard compatibility |
| Particle initialization | New IC generation | Existing `_build_particle()` pattern | Already handles device/dtype correctly |
| Save directory construction | New path logic | Existing `data/{save_name}/model/` pattern | Consistent with current checkpoints |

**Key insight:** The integration should change as little as possible outside the training loop. All infrastructure code (device setup, seeds, particle init, model creation) is already correct.

## Common Pitfalls

### Pitfall 1: Breaking Multi-Orbit Support
**What goes wrong:** New training ignores `num_orbits > 1`, users get silent degradation.
**Why it happens:** New functions only support single particle, not batched.
**How to avoid:** Add explicit check and warning when `num_orbits > 1`:
```python
if config.num_orbits > 1:
    warnings.warn(
        f"Multi-orbit training (num_orbits={config.num_orbits}) not yet supported "
        "in two-phase training. Using single orbit. For batched training, use legacy mode.",
        UserWarning,
    )
```
**Warning signs:** User passes `--num-orbits 8` but only gets single-orbit behavior.

### Pitfall 2: History Buffer Ownership
**What goes wrong:** Creating new HistoryBuffer in run_training() instead of using adapter's.
**Why it happens:** Copy-paste from multi-orbit case which creates separate buffers.
**How to avoid:** Always use `adapter.history_buffer` which is already correctly initialized:
```python
# CORRECT:
history_buffer=adapter.history_buffer

# WRONG:
history_buffer=HistoryBuffer(...)  # Creates disconnected buffer
```
**Warning signs:** History features not matching expected dimensions.

### Pitfall 3: W&B Run Passed vs Module Import
**What goes wrong:** `run_two_phase_training()` receives wandb_run but can't call `wandb.log()`.
**Why it happens:** The run object and the module are separate; function needs module for logging.
**How to avoid:** Already handled - `run_two_phase_training()` imports wandb internally if wandb_run is not None.
**Warning signs:** W&B logging disabled despite passing wandb_run.

### Pitfall 4: Save Directory Not Matching Expectation
**What goes wrong:** Checkpoints saved to different location than previous system.
**Why it happens:** Using different path construction logic.
**How to avoid:** Match exact pattern from current runner.py:
```python
save_dir = pathlib.Path(project_root) / "data" / (config.save_name or "run_nbody") / "model"
```
**Warning signs:** Simulation mode can't find checkpoints.

### Pitfall 5: Result Not Logged After Training
**What goes wrong:** Training completes silently, user doesn't know outcome.
**Why it happens:** Old code printed `epoch N loss=X saved=Y`; new code returns dict.
**How to avoid:** Add summary print after `run_two_phase_training()` returns:
```python
result = run_two_phase_training(...)
print(f"Training complete: {result['epochs_completed']} epochs, "
      f"convergence rate: {result['convergence_rate']:.1%}")
```
**Warning signs:** Training finishes with no output (only W&B logs).

## Import Changes

### Current imports in runner.py:

```python
from src import (
    Config,
    FullyConnectedNN,
    HistoryBuffer,
    ModelAdapter,
    load_model_state,
    load_config_from_checkpoint,
    loss_fn_batch,                    # REMOVE - not used in new system
    loss_fn_batch_history,            # REMOVE - not used in new system
    loss_fn_batch_history_batch,      # REMOVE - not used in new system
    make_particle,
    save_checkpoint,                   # KEEP - still used for path construction
    stack_particles,                   # KEEP - may be used later for multi-orbit
)
```

### New imports needed:

```python
from src import (
    # ... existing kept imports ...
    run_two_phase_training,           # ADD - new training function
)
```

### Functions that become unused (safe to remove):

- `loss_fn_batch` - replaced by internal calls in collect_trajectory/generalize
- `loss_fn_batch_history` - replaced by internal calls
- `loss_fn_batch_history_batch` - multi-orbit version, not used yet

## Code Examples

### Complete Refactored run_training()

```python
def run_training(config: Config) -> None:
    """Train ML time-stepper using two-phase training system."""

    # === Validation ===
    if config.integrator_mode == "history" and (config.history_len is None or config.history_len <= 0):
        raise ValueError("history integrator_mode requires history_len > 0")
    config.validate()

    # === Device/dtype setup ===
    device = config.resolve_device()
    dtype = config.resolve_dtype()
    torch.set_default_dtype(dtype)
    config.apply_torch_settings(device)

    # === Seed ===
    if config.seed is not None:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

    # === Model adapter ===
    adapter = ModelAdapter(config, device=device, dtype=dtype)

    # === W&B setup ===
    wandb_run = None
    wandb = None
    if config.extra.get("wandb", False):
        try:
            import wandb as wandb_lib
        except ImportError as exc:
            raise RuntimeError("wandb is not installed; install it or disable --wandb") from exc
        wandb = wandb_lib
        wandb_project = config.extra.get("wandb_project") or "AITimeStepper"
        wandb_name = config.extra.get("wandb_name") or config.save_name or "runner_train"
        wandb_run = wandb.init(
            project=wandb_project,
            name=wandb_name,
            config=config.as_wandb_dict(),
        )

    # === Multi-orbit check (temporary limitation) ===
    if config.num_orbits > 1:
        import warnings
        warnings.warn(
            f"Multi-orbit training (num_orbits={config.num_orbits}) not yet supported "
            "in two-phase training. Using single orbit.",
            UserWarning,
        )

    # === Particle initialization ===
    ptcls = generate_random_ic(
        num_particles=config.num_particles,
        dim=config.dim,
        mass=config.mass,
        pos_scale=config.pos_scale,
        vel_scale=config.vel_scale,
        seed=config.seed,
    )
    particle = make_particle(ptcls, device=device, dtype=dtype)
    particle.current_time = torch.tensor(0.0, device=device, dtype=dtype)

    # === Model creation ===
    input_dim = adapter.input_dim_from_state(particle, history_buffer=adapter.history_buffer)
    model = FullyConnectedNN(
        input_dim=input_dim,
        output_dim=2,
        hidden_dims=[200, 1000, 1000, 200],
        activation="tanh",
        dropout=0.2,
        output_positive=True,
    ).to(device)
    model.to(dtype=dtype)

    # === Optimizer ===
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # === Training (new two-phase system) ===
    save_dir = pathlib.Path(project_root) / "data" / (config.save_name or "run_nbody") / "model"

    result = run_two_phase_training(
        model=model,
        particle=particle,
        optimizer=optimizer,
        config=config,
        adapter=adapter,
        history_buffer=adapter.history_buffer,
        save_dir=save_dir,
        wandb_run=wandb_run,
        checkpoint_interval=10,
    )

    # === Summary ===
    print(f"Training complete: {result['epochs_completed']} epochs, "
          f"convergence rate: {result['convergence_rate']:.1%}, "
          f"total time: {result['total_time']:.1f}s")

    # === W&B cleanup ===
    if wandb_run is not None:
        wandb.finish()
```

### Minimal Integration (Just Replace Loop)

If the full refactor is too aggressive, the minimal change is:

```python
# In run_training(), replace lines 313-391 (the training loop) with:

save_dir = pathlib.Path(project_root) / "data" / (config.save_name or "run_nbody") / "model"

# Use the particle from single-orbit case only
if config.num_orbits > 1:
    warnings.warn("Multi-orbit not supported in new training, using single orbit")
    particle = particles[0]  # From the multi-orbit initialization

result = run_two_phase_training(
    model=model,
    particle=particle,
    optimizer=optimizer,
    config=config,
    adapter=adapter,
    history_buffer=adapter.history_buffer if not histories else histories[0],
    save_dir=save_dir,
    wandb_run=wandb_run,
    checkpoint_interval=10,
)

# Keep the W&B finish call unchanged
```

## State of the Art

| Old Approach (runner.py) | New Approach (Phase 6) | Impact |
|--------------------------|------------------------|--------|
| Multi-step loss per epoch | Two-phase: collect + generalize | Better energy conservation |
| `loss_fn_batch*` functions | `collect_trajectory` + `generalize_on_trajectory` | Quality-gated samples |
| Manual checkpoint loop | `run_two_phase_training` handles it | Cleaner code |
| 4 different loss functions | Single internal loss computation | Less code duplication |

**Deprecated after Phase 6:**
- `loss_fn_batch` - replaced by trajectory collection
- `loss_fn_batch_history` - replaced by trajectory collection with buffer
- `loss_fn_batch_history_batch` - deferred (multi-orbit not in scope)

## Open Questions

### Q1: What happens to n_steps parameter?
- **What we know:** Old system used `config.n_steps` for multi-step integration per loss
- **What's unclear:** Is n_steps still meaningful in two-phase?
- **Recommendation:** n_steps is no longer used in two-phase training (single-step only). Keep in Config for backward compatibility but document as legacy. Users should use `steps_per_epoch` instead.

### Q2: Should we remove the old loss function imports?
- **What we know:** They're no longer called in run_training()
- **What's unclear:** Are they used anywhere else?
- **Recommendation:** Keep imports but add comment marking them as legacy. Remove in Phase 7 (cleanup) after verifying no other usage.

### Q3: How to handle --duration flag?
- **What we know:** Old training had `if config.duration is not None and (time.perf_counter() - start_time) >= config.duration: break`
- **What's unclear:** Does `run_two_phase_training()` respect duration?
- **Recommendation:** Duration is not currently supported in run_two_phase_training(). For Phase 6, add warning if duration is set. Implement duration support as enhancement if needed.

## Testing Strategy

### Test 1: Basic CLI Compatibility
```bash
# These commands should still work:
python run/runner.py train --epochs 10 --n-steps 5 --num-particles 4 --save-name test_phase6

# With history:
python run/runner.py train --epochs 10 --history-len 5 --feature-type delta_mag --num-particles 4 --save-name test_phase6_hist
```

### Test 2: Checkpoint Contract
```bash
# Train, then load in simulation:
python run/runner.py train --epochs 10 --num-particles 4 --save-name test_compat
python run/runner.py simulate --integrator-mode ml --model-path data/test_compat/model/model_epoch_0009.pt --num-particles 4 --steps 100
```

### Test 3: W&B Logging
```bash
# Verify metrics appear in dashboard:
python run/runner.py train --epochs 10 --num-particles 4 --wandb --wandb-project AITimeStepper_test
# Check for part1/*, part2/* metric groups
```

### Test 4: Multi-Orbit Warning
```bash
# Should warn and proceed with single orbit:
python run/runner.py train --epochs 10 --num-orbits 8 --num-particles 4 --save-name test_multi
# Expect warning message in output
```

## Sources

### Primary (HIGH confidence)
- `/u/gkerex/projects/AITimeStepper/run/runner.py` - Current implementation analyzed
- `/u/gkerex/projects/AITimeStepper/src/unified_training.py` - New training functions verified
- `/u/gkerex/projects/AITimeStepper/src/checkpoint.py` - Checkpoint contract verified
- `/u/gkerex/projects/AITimeStepper/src/config.py` - Config fields verified
- `/u/gkerex/projects/AITimeStepper/src/model_adapter.py` - Adapter usage verified

### Secondary (MEDIUM confidence)
- `/u/gkerex/projects/AITimeStepper/.planning/phases/05-unified-epoch-structure/05-RESEARCH.md` - Phase 5 patterns

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All functions exist and signatures verified
- Architecture: HIGH - Direct integration with minimal changes
- Pitfalls: HIGH - Edge cases identified from code analysis

**Research date:** 2026-01-21
**Valid until:** 2026-02-21 (30 days - stable domain, internal codebase)
