# Phase 5: Unified Epoch Structure - Research

**Researched:** 2026-01-21
**Domain:** Training orchestration / PyTorch training loops
**Confidence:** HIGH

## Summary

Phase 5 combines the completed Part 1 (trajectory collection) and Part 2 (generalization training) into a unified epoch structure. The research examined the exact function signatures from Phase 3 and Phase 4, the checkpoint/logging patterns in `runner.py`, and the Config system to understand how to orchestrate the two-phase training loop.

The implementation is straightforward: `train_epoch_two_phase()` will call `collect_trajectory()` then `generalize_on_trajectory()`, wrapping this in an outer loop for `config.epochs` iterations. The existing checkpoint and W&B patterns from `runner.py` provide clear templates.

**Primary recommendation:** Create a new module `src/unified_training.py` with `train_epoch_two_phase()` that combines the existing functions, using the established checkpoint pattern at intervals of 10 epochs or on final epoch.

## Standard Stack

### Core Functions (Already Implemented)

| Function | Location | Purpose | Signature |
|----------|----------|---------|-----------|
| `collect_trajectory` | `src/trajectory_collection.py` | Part 1: N validated steps | `(model, particle, optimizer, config, adapter, history_buffer?) -> (trajectory, epoch_metrics)` |
| `generalize_on_trajectory` | `src/generalization_training.py` | Part 2: Convergence loop | `(model, trajectory, optimizer, config, adapter, history_buffer?) -> (converged, iterations, final_metrics)` |
| `save_checkpoint` | `src/checkpoint.py` | Save model state | `(path, model, optimizer?, epoch?, loss?, info?, logs?, config?, extra?) -> Path` |

### Supporting Infrastructure

| Component | Location | Purpose |
|-----------|----------|---------|
| `Config` | `src/config.py` | All parameters including `epochs`, `energy_threshold`, `steps_per_epoch` |
| `ModelAdapter` | `src/model_adapter.py` | Feature construction abstraction |
| `HistoryBuffer` | `src/history_buffer.py` | Temporal state tracking |

### W&B Integration

| Function/Method | Usage Pattern |
|----------------|---------------|
| `wandb.init()` | Initialize run at training start |
| `wandb.log()` | Log metrics per epoch |
| `wandb.finish()` | Cleanup at training end |
| `config.as_wandb_dict()` | Serialize config for W&B |

## Architecture Patterns

### Function Signatures (Verified from Codebase)

**collect_trajectory() signature:**
```python
def collect_trajectory(
    model: torch.nn.Module,
    particle: ParticleTorch,
    optimizer: torch.optim.Optimizer,
    config: Config,
    adapter: ModelAdapter,
    history_buffer: Optional[HistoryBuffer] = None,
) -> Tuple[List[Tuple[ParticleTorch, float]], Dict[str, Any]]:
    """
    Returns:
        (trajectory, epoch_metrics) where:
        - trajectory: List of (ParticleTorch, dt) tuples (warmup excluded)
        - epoch_metrics: dict with:
            'total_steps', 'warmup_discarded', 'trajectory_length',
            'mean_retrain_iterations', 'mean_energy_error', 'max_retrain_iterations'
    """
```

**generalize_on_trajectory() signature:**
```python
def generalize_on_trajectory(
    model: torch.nn.Module,
    trajectory: List[Tuple[ParticleTorch, float]],
    optimizer: torch.optim.Optimizer,
    config: Config,
    adapter: ModelAdapter,
    history_buffer: Optional[HistoryBuffer] = None,
) -> Tuple[bool, int, Dict[str, Any]]:
    """
    Returns:
        (converged, iterations, final_metrics) where:
        - converged: True if all samples passed before max iterations
        - iterations: Number of iterations completed
        - final_metrics: dict with:
            'mean_rel_dE', 'max_rel_dE', 'final_pass_rate',
            optionally 'skipped' if trajectory was too small
    """
```

### Checkpoint Pattern (from runner.py)

```python
# Save every 10 epochs or on final epoch
if epoch % 10 == 0 or epoch == config.epochs - 1:
    save_dir = pathlib.Path(project_root) / "data" / (config.save_name or "run_nbody") / "model"
    save_path = save_dir / f"model_epoch_{epoch:04d}.pt"
    save_checkpoint(
        save_path,
        model,
        optimizer,
        epoch=epoch,
        loss=loss,
        logs=logs,
        config=config,
    )
```

### W&B Logging Pattern (from runner.py)

```python
# Initialize
wandb_run = wandb.init(
    project=wandb_project,
    name=wandb_name,
    config=config.as_wandb_dict(),
)

# Per-epoch logging
wandb.log({
    "epoch": epoch,
    "loss": float(loss.item()),
    # ... additional metrics
})

# Cleanup
wandb.finish()
```

### Proposed train_epoch_two_phase() Structure

```python
def train_epoch_two_phase(
    model: torch.nn.Module,
    particle: ParticleTorch,
    optimizer: torch.optim.Optimizer,
    config: Config,
    adapter: ModelAdapter,
    history_buffer: Optional[HistoryBuffer] = None,
) -> Dict[str, Any]:
    """
    Execute one complete epoch: Part 1 (collection) + Part 2 (generalization).

    Returns:
        epoch_result: dict with:
            'trajectory_metrics': from Part 1
            'generalization_metrics': from Part 2
            'converged': bool from Part 2
            'part2_iterations': int from Part 2
    """
    # Part 1: Collect trajectory
    trajectory, traj_metrics = collect_trajectory(
        model, particle, optimizer, config, adapter, history_buffer
    )

    # Part 2: Generalize on trajectory
    converged, part2_iters, gen_metrics = generalize_on_trajectory(
        model, trajectory, optimizer, config, adapter, history_buffer
    )

    return {
        'trajectory_metrics': traj_metrics,
        'generalization_metrics': gen_metrics,
        'converged': converged,
        'part2_iterations': part2_iters,
    }
```

### Proposed run_two_phase_training() Structure

```python
def run_two_phase_training(
    model: torch.nn.Module,
    particle: ParticleTorch,
    optimizer: torch.optim.Optimizer,
    config: Config,
    adapter: ModelAdapter,
    history_buffer: Optional[HistoryBuffer] = None,
    wandb_run: Optional[Any] = None,
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Execute N epochs of two-phase training with checkpointing and logging.

    Returns:
        training_result: dict with aggregated metrics across all epochs
    """
    for epoch in range(config.epochs):
        epoch_result = train_epoch_two_phase(
            model, particle, optimizer, config, adapter, history_buffer
        )

        # W&B logging
        if wandb_run is not None:
            wandb.log({...})

        # Checkpointing
        if epoch % 10 == 0 or epoch == config.epochs - 1:
            save_checkpoint(...)

    return final_results
```

## Config Parameters (Verified)

From `src/config.py`, the relevant fields for Phase 5:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `epochs` | int | 1000 | Total number of epochs to run |
| `energy_threshold` | float | 2e-4 | Accept/reject threshold |
| `steps_per_epoch` | int | 1 | N steps per Part 1 collection |
| `replay_steps` | int | 1000 | Max iterations for Part 2 |
| `replay_batch_size` | int | 512 | Minibatch size for Part 2 |
| `min_replay_size` | int | 2 | Skip Part 2 if trajectory smaller |
| `history_len` | int | 0 | History buffer length |
| `save_name` | str | None | Base directory for saves |

**Validation (already in Config.validate()):**
- `epochs >= 1`
- `steps_per_epoch >= 1`
- `energy_threshold > 0`

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Checkpoint serialization | Custom save format | `save_checkpoint()` | Already handles Config, optimizer, epoch, etc. |
| W&B config serialization | Custom dict builder | `config.as_wandb_dict()` | Handles extra fields, type conversion |
| Feature construction | Direct feature code | `ModelAdapter` | Abstracts history/analytic modes |
| Particle cloning | Manual tensor copies | `particle.clone_detached()` | Ensures no graph accumulation |

**Key insight:** All infrastructure exists. Phase 5 is pure orchestration of existing functions.

## Common Pitfalls

### Pitfall 1: Graph Accumulation Across Epochs
**What goes wrong:** Keeping particles in computation graph causes OOM over epochs.
**Why it happens:** Returning raw particles from Part 1 to Part 2 without detaching.
**How to avoid:** Both `collect_trajectory()` and `generalize_on_trajectory()` already use `clone_detached()` internally. No additional work needed.
**Warning signs:** Memory usage growing linearly with epochs.

### Pitfall 2: Part 2 Fails to Converge
**What goes wrong:** Part 2 never converges, epoch takes very long.
**Why it happens:** Energy threshold too strict, model poorly initialized.
**How to avoid:** Already mitigated - `generalize_on_trajectory()` has `config.replay_steps` as safety limit. Epoch completes after max iterations even if not converged.
**Warning signs:** `converged=False` frequently, `part2_iterations==replay_steps` every epoch.

### Pitfall 3: Empty Trajectory Edge Case
**What goes wrong:** Part 1 returns empty trajectory, Part 2 does nothing.
**Why it happens:** `steps_per_epoch <= history_len` causes all steps to be warmup.
**How to avoid:** Already handled - `generalize_on_trajectory()` returns `(True, 0, ...)` for empty trajectory. Config.validate() already enforces `steps_per_epoch >= 1`. Add warning log if trajectory empty.
**Warning signs:** `trajectory_length==0` in metrics.

### Pitfall 4: History Buffer Reset Between Epochs
**What goes wrong:** History buffer cleared at epoch end, losing context.
**Why it happens:** Creating new buffer each epoch.
**How to avoid:** Pass same `history_buffer` instance across all epochs in outer loop.
**Warning signs:** First `history_len` steps always in warmup mode.

### Pitfall 5: Particle State Drift Between Epochs
**What goes wrong:** Using evolved particle as initial condition causes drift.
**Why it happens:** Part 1 advances particle state; using final state for next epoch.
**How to avoid:** For energy conservation testing, start each epoch from fixed initial conditions. For trajectory generation, this may be intentional behavior.
**Warning signs:** Energy systematically drifting over epochs.

## W&B Metrics to Log

Based on existing runner.py patterns and new two-phase structure:

### Per-Epoch Metrics

| Metric | Source | Type |
|--------|--------|------|
| `epoch` | Loop counter | int |
| `part1/trajectory_length` | `traj_metrics['trajectory_length']` | int |
| `part1/mean_retrain_iterations` | `traj_metrics['mean_retrain_iterations']` | float |
| `part1/mean_energy_error` | `traj_metrics['mean_energy_error']` | float |
| `part1/max_retrain_iterations` | `traj_metrics['max_retrain_iterations']` | int |
| `part1/warmup_discarded` | `traj_metrics['warmup_discarded']` | int |
| `part2/converged` | `converged` | bool (0/1) |
| `part2/iterations` | `part2_iters` | int |
| `part2/final_pass_rate` | `gen_metrics['final_pass_rate']` | float |
| `part2/mean_rel_dE` | `gen_metrics['mean_rel_dE']` | float |
| `part2/max_rel_dE` | `gen_metrics['max_rel_dE']` | float |

### Derived Metrics (Optional)

| Metric | Calculation | Purpose |
|--------|-------------|---------|
| `part1/acceptance_rate` | `1 / (1 + mean_retrain_iterations)` | Measure of model quality |
| `epoch_time` | `time.perf_counter()` delta | Training speed tracking |

## Code Examples

### Complete Epoch Orchestration

```python
def train_epoch_two_phase(
    model: torch.nn.Module,
    particle: ParticleTorch,
    optimizer: torch.optim.Optimizer,
    config: Config,
    adapter: ModelAdapter,
    history_buffer: Optional[HistoryBuffer] = None,
) -> Dict[str, Any]:
    """Execute one complete epoch: Part 1 + Part 2."""

    # Part 1: Collect trajectory (TRAIN-04)
    trajectory, traj_metrics = collect_trajectory(
        model=model,
        particle=particle,
        optimizer=optimizer,
        config=config,
        adapter=adapter,
        history_buffer=history_buffer,
    )

    # Part 2: Generalize on trajectory (TRAIN-05, TRAIN-06)
    converged, part2_iters, gen_metrics = generalize_on_trajectory(
        model=model,
        trajectory=trajectory,
        optimizer=optimizer,
        config=config,
        adapter=adapter,
        history_buffer=history_buffer,
    )

    return {
        'trajectory_metrics': traj_metrics,
        'generalization_metrics': gen_metrics,
        'converged': converged,
        'part2_iterations': part2_iters,
    }
```

### Outer Training Loop with Checkpointing

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
) -> Dict[str, Any]:
    """Run N epochs of two-phase training (TRAIN-08, TRAIN-09)."""

    all_results = []

    for epoch in range(config.epochs):
        # Execute one epoch
        epoch_result = train_epoch_two_phase(
            model, particle, optimizer, config, adapter, history_buffer
        )
        all_results.append(epoch_result)

        # W&B logging
        if wandb_run is not None:
            traj_m = epoch_result['trajectory_metrics']
            gen_m = epoch_result['generalization_metrics']
            wandb.log({
                'epoch': epoch,
                'part1/trajectory_length': traj_m['trajectory_length'],
                'part1/mean_retrain_iterations': traj_m['mean_retrain_iterations'],
                'part2/converged': int(epoch_result['converged']),
                'part2/iterations': epoch_result['part2_iterations'],
                'part2/final_pass_rate': gen_m['final_pass_rate'],
            })

        # Checkpoint every 10 epochs or final
        if save_dir and (epoch % 10 == 0 or epoch == config.epochs - 1):
            save_path = save_dir / f"model_epoch_{epoch:04d}.pt"
            save_checkpoint(
                save_path,
                model,
                optimizer,
                epoch=epoch,
                logs=epoch_result,
                config=config,
            )

    return {'epochs_completed': config.epochs, 'results': all_results}
```

## State of the Art

| Old Approach (runner.py) | New Approach (Phase 5) | Impact |
|--------------------------|------------------------|--------|
| Multi-step loss per epoch | Single-step collection + generalization | Better energy conservation |
| Fixed trajectory length | Accept/reject with retrain loop | Quality-gated samples |
| Loss computed on trajectory | Convergence criterion on minibatches | Guaranteed generalization |

**Key change:** The old approach computed loss over N steps without quality gates. The new approach ensures every accepted sample satisfies energy threshold, then trains until generalization converges.

## Open Questions

### Q1: Should particle state reset between epochs?
- **What we know:** Part 1 evolves particle forward; Part 2 replays trajectory
- **What's unclear:** Should epoch N+1 start from evolved state or original IC?
- **Recommendation:** Start from original IC for reproducibility. Document as design decision. Can make configurable if needed.

### Q2: History buffer handling across epochs
- **What we know:** History buffer accumulates state during Part 1
- **What's unclear:** Should buffer persist across epochs or reset?
- **Recommendation:** Persist buffer across epochs (same instance). Buffer already handles warmup internally. This matches "continuing trajectory" semantics.

### Q3: What happens if Part 1 produces empty trajectory?
- **What we know:** `generalize_on_trajectory()` handles empty trajectory (returns converged=True, iter=0)
- **What's unclear:** Should we log a warning? Skip the epoch entirely?
- **Recommendation:** Log a warning, but continue. Empty trajectory is valid edge case when `steps_per_epoch <= history_len`.

## Sources

### Primary (HIGH confidence)
- `src/trajectory_collection.py` - Function signatures and docstrings verified
- `src/generalization_training.py` - Function signatures and docstrings verified
- `src/config.py` - Config fields and validation verified
- `src/checkpoint.py` - save_checkpoint signature verified
- `run/runner.py` - Checkpoint and W&B patterns verified

### Secondary (MEDIUM confidence)
- `tests/test_trajectory_collection.py` - Test patterns for mocking
- `tests/test_generalization_training.py` - Test patterns for verification

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All functions exist and verified
- Architecture: HIGH - Direct composition of existing functions
- Pitfalls: HIGH - Edge cases already handled by Phase 3/4 implementations

**Research date:** 2026-01-21
**Valid until:** 2026-02-21 (30 days - stable domain)
