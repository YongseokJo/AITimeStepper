# Phase 3: Trajectory Collection Loop - Research

**Researched:** 2026-01-20
**Domain:** PyTorch training loop, differentiable physics integration, accept/reject sampling
**Confidence:** HIGH

## Summary

Phase 3 implements iterative trajectory collection with energy-based quality gates. The core pattern is: predict dt → integrate one step → check energy conservation → if pass, accept; if fail, retrain on same state until passing.

This research examined the existing training infrastructure in AITimeStepper to understand:
1. How the current training loop works (multi-step forward pass with batched loss)
2. How energy is computed and checked (ParticleTorch.total_energy methods)
3. How integration works (evolve/evolve_batch with leapfrog)
4. Optimizer step patterns (standard PyTorch Adam)
5. Where trajectory collection should live (new function in runner.py or separate module)

**Primary recommendation:** Create a new `collect_trajectory_step()` function that accepts particles, model, optimizer, config, and history buffer, returns accepted (state, dt) pairs. Place in runner.py initially for Phase 3-5, then refactor to separate trainer module in Phase 6.

## Standard Stack

The codebase already uses the standard PyTorch training stack. No new dependencies needed.

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | (existing) | Autograd, optimization, differentiable physics | Already core dependency |
| NumPy | (existing) | Initial conditions, non-training simulation | Standard scientific Python |

### Supporting
No new libraries needed for Phase 3. The following existing patterns are sufficient:

| Pattern | Location | Purpose |
|---------|----------|---------|
| `optimizer.zero_grad()` + `loss.backward()` + `optimizer.step()` | runner.py lines 366-368 | Standard gradient update |
| `particle.clone_detached()` | particle.py lines 87-103 | Create fresh copy for forward pass |
| `particle.total_energy()` / `total_energy_batch()` | particle.py lines 286-428 | Energy computation |
| `particle.evolve()` / `evolve_batch()` | particle.py lines 178-284 | Leapfrog integration |
| `ModelAdapter.build_feature_tensor()` | model_adapter.py lines 38-55 | Feature extraction (analytic or history) |

## Architecture Patterns

### Recommended Function Structure

Phase 3 introduces a new training primitive: **accept/reject step with retrain loop**.

```python
# In runner.py or new src/trainer_two_phase.py

def collect_trajectory_step(
    model: torch.nn.Module,
    particle: ParticleTorch,
    optimizer: torch.optim.Optimizer,
    config: Config,
    adapter: ModelAdapter,
    history_buffer: Optional[HistoryBuffer] = None,
) -> Tuple[ParticleTorch, float, Dict[str, Any]]:
    """
    Collect one accepted trajectory step with retrain loop.

    Returns:
        (accepted_particle, accepted_dt, metrics)

    Metrics includes:
        - 'retrain_iterations': number of optimizer steps needed
        - 'final_energy_error': relative energy error of accepted step
        - 'reject_count': number of rejected attempts
    """
    pass
```

### Pattern 1: Accept/Reject Loop with Local Optimizer Steps

**What:** Repeatedly attempt a step, retraining until energy threshold satisfied

**When to use:** Part 1 trajectory collection (Phase 3)

**Example:**
```python
# Source: Designed for AITimeStepper Phase 3 based on existing patterns

accept_threshold = config.energy_threshold  # e.g., 2e-4
retrain_iterations = 0

while True:
    # 1. Fresh copy for this attempt
    p_attempt = particle.clone_detached()

    # 2. Build features (analytic or history-aware)
    feats = adapter.build_feature_tensor(p_attempt, history_buffer=history_buffer)
    if feats.dim() == 1:
        feats = feats.unsqueeze(0)

    # 3. Predict dt
    params = model(feats)
    dt_raw = params[:, 0]
    dt = dt_raw + 1e-12  # positive constraint

    # 4. Compute initial energy
    E0 = p_attempt.total_energy_batch(G=1.0)
    if E0.dim() == 0:
        E0 = E0.unsqueeze(0)

    # 5. Integrate one step
    p_attempt.update_dt(dt)
    p_attempt.evolve_batch(G=1.0)

    # 6. Compute final energy
    E1 = p_attempt.total_energy_batch(G=1.0)
    if E1.dim() == 0:
        E1 = E1.unsqueeze(0)

    # 7. Check energy threshold
    E0_safe = E0 + 1e-12 * E0.detach().abs() + 1e-12
    rel_dE = torch.abs((E1 - E0) / E0_safe)

    if rel_dE.item() < accept_threshold:
        # ACCEPT: energy within threshold
        return p_attempt, dt.item(), {'retrain_iterations': retrain_iterations, 'final_energy_error': rel_dE.item()}

    # REJECT: retrain on same state
    # Use existing loss function pattern
    loss = compute_loss_for_step(p_attempt, E0, E1, config)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    retrain_iterations += 1
```

### Pattern 2: Trajectory Buffer (Simple List)

**What:** Store accepted (state, dt) pairs for Part 2 training

**When to use:** Part 1 output, Part 2 input

**Example:**
```python
# Source: Standard Python pattern, adapted for ParticleTorch

trajectory = []  # List[Tuple[ParticleTorch, float]]

for _ in range(config.steps_per_epoch):
    accepted_particle, accepted_dt, metrics = collect_trajectory_step(...)

    # Store detached copy to avoid graph accumulation
    state_copy = accepted_particle.clone_detached()
    trajectory.append((state_copy, accepted_dt))

    # Update history buffer if enabled
    if history_buffer is not None:
        history_buffer.push(state_copy)
```

### Pattern 3: Warmup Discard (HIST-02)

**What:** Discard first `history_len` steps once buffer fills, only keep real trajectory

**When to use:** History-enabled training after initial bootstrap

**Example:**
```python
# Source: Designed for HIST-02 requirement

def collect_trajectory(
    model, particle, optimizer, config, adapter, history_buffer
) -> List[Tuple[ParticleTorch, float]]:
    trajectory = []

    for step_idx in range(config.steps_per_epoch):
        accepted_particle, accepted_dt, metrics = collect_trajectory_step(...)

        # Always push to history (needed for features)
        if history_buffer is not None:
            history_buffer.push(accepted_particle)

        # Discard warmup steps (first history_len)
        if history_buffer is not None and step_idx < config.history_len:
            # Warmup: don't add to trajectory, but do update history
            continue

        # Real trajectory: add to buffer
        trajectory.append((accepted_particle, accepted_dt))

    return trajectory
```

### Anti-Patterns to Avoid

- **Reusing same particle object without clone_detached():** Breaks autograd graph, causes memory leaks
- **Checking E1 - E0 instead of (E1 - E0) / E0:** Absolute error fails for different energy scales
- **Using multi-step loss during retrain loop:** Part 1 is single-step only (TRAIN-01)
- **Infinite loop without diagnostics:** Add iteration counter and optional max_retrain_iterations for debugging

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Feature extraction | Custom history concatenation | `ModelAdapter.build_feature_tensor()` | Handles both analytic and history modes, batching, device placement |
| Energy computation | Manual kinetic + potential | `particle.total_energy_batch()` | Handles batching, softening, external fields, safe division |
| Integration | Manual Euler or RK4 | `particle.evolve_batch()` | Symplectic leapfrog preserves conservation, supports batching |
| State copying | Manual tensor clone loop | `particle.clone_detached()` | Properly detaches all tensors, preserves external_field |
| Loss computation | New loss function | Existing `band_loss_zero_inside_where` | Already implements tolerance band logic |

**Key insight:** The codebase already has differentiable physics primitives. Phase 3 is about orchestrating them in a new control flow pattern (accept/reject loop), not reimplementing physics.

## Common Pitfalls

### Pitfall 1: Graph Accumulation During Retrain Loop
**What goes wrong:** Each retrain iteration keeps the computation graph alive, causing OOM after many iterations

**Why it happens:** PyTorch retains intermediate tensors for backward pass. Repeated backward() calls accumulate graphs.

**How to avoid:**
- Clone particle state with `clone_detached()` at start of each attempt
- Don't store accepted particles in a list during loop (only after acceptance)
- Use `optimizer.zero_grad()` before each backward pass

**Warning signs:** Memory usage grows with retrain_iterations, OOM after ~100 retrain steps

### Pitfall 2: Threshold Check on Wrong Energy Quantity
**What goes wrong:** Checking absolute energy error (E1 - E0) instead of relative error ((E1 - E0) / E0)

**Why it happens:** Existing loss functions use log-space relative errors, easy to confuse with absolute

**How to avoid:** Always compute `rel_dE = abs((E1 - E0) / E0_safe)` and compare to threshold

**Warning signs:** Threshold never satisfied even with small dt, or satisfied trivially with large dt

### Pitfall 3: Using Multi-Step Integration During Part 1
**What goes wrong:** Calling loss functions with `n_steps > 1` during trajectory collection

**Why it happens:** Existing training uses `config.n_steps = 10` for multi-step rollout

**How to avoid:** Part 1 always uses single-step integration (TRAIN-01). Save n_steps > 1 for Part 2 (Phase 4).

**Warning signs:** Trajectory buffer has gaps, history buffer sees large dt jumps

### Pitfall 4: Discarding Warmup Steps Incorrectly
**What goes wrong:** Not pushing warmup steps to history buffer, or discarding non-warmup steps

**Why it happens:** HIST-02 says "discard warmup steps" - easy to skip history.push() entirely

**How to avoid:**
- Always push to history buffer (needed for next step's features)
- Only skip appending to trajectory list for first history_len steps
- Check: `if step_idx < config.history_len: continue` after history.push()

**Warning signs:** History buffer empty after warmup, features_for() crashes, zero-padding persists beyond warmup

### Pitfall 5: Loss Function Mismatch (Analytic vs History)
**What goes wrong:** Calling `loss_fn_batch()` (analytic) when history enabled, or vice versa

**Why it happens:** Two separate loss modules: losses.py (analytic) and losses_history.py (history)

**How to avoid:**
```python
if adapter.history_enabled:
    from src.losses_history import loss_fn_batch_history
    loss, logs, _ = loss_fn_batch_history(model, particle, history_buffer, ...)
else:
    from src.losses import loss_fn_batch
    loss, logs, _ = loss_fn_batch(model, particle, ...)
```

**Warning signs:** Feature dimension mismatch error, KeyError for history-specific metrics

## Code Examples

Verified patterns from existing codebase:

### Current Training Loop (Multi-Step, For Comparison)
```python
# Source: runner.py lines 315-391
# This is what Phase 3 REPLACES for Part 1

while epoch < config.epochs:
    if adapter.history_enabled:
        loss, logs, _ = loss_fn_batch_history(
            model,
            particle,
            adapter.history_buffer,
            n_steps=config.n_steps,  # Multi-step (e.g., 10)
            rel_loss_bound=config.rel_loss_bound,
            E_lower=config.E_lower,
            E_upper=config.E_upper,
            L_lower=config.L_lower,
            L_upper=config.L_upper,
            return_particle=True,
        )
    else:
        loss, logs = loss_fn_batch(
            model,
            particle,
            n_steps=config.n_steps,
            rel_loss_bound=config.rel_loss_bound,
            E_lower=config.E_lower,
            E_upper=config.E_upper,
        )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # One optimizer step per epoch
    epoch += 1
```

**Key difference for Phase 3:** Part 1 does single-step integration with multiple optimizer steps per accepted step (retrain loop), not multi-step integration with one optimizer step.

### Energy Computation (Existing Pattern)
```python
# Source: particle.py lines 360-428
# Already handles batching correctly

E0 = particle.total_energy_batch(G=1.0)  # Returns (B,) or scalar
if E0.dim() == 0:
    E0 = E0.unsqueeze(0)  # Normalize to (1,)

# After integration
E1 = particle.total_energy_batch(G=1.0)
if E1.dim() == 0:
    E1 = E1.unsqueeze(0)

# Safe relative error (handles division by zero)
eps = 1e-12
E0_safe = E0 + eps * E0.detach().abs() + eps
rel_dE = torch.abs((E1 - E0) / E0_safe)
```

### Loss Computation for Single Step (Needed for Retrain Loop)
```python
# Source: Adapted from losses.py lines 153-310
# Phase 3 needs a simplified single-step loss for retrain loop

def compute_single_step_loss(
    particle: ParticleTorch,
    E0: torch.Tensor,
    E1: torch.Tensor,
    config: Config,
) -> torch.Tensor:
    """
    Simplified loss for accept/reject retrain loop.
    Only penalizes energy error outside tolerance band.
    """
    eps = 1e-12
    E0_safe = E0 + eps * E0.detach().abs() + eps
    rel_dE = torch.abs((E1 - E0) / E0_safe)

    # Replace inf/nan with large penalty
    rel_dE = torch.where(
        torch.isfinite(rel_dE),
        rel_dE,
        torch.full_like(rel_dE, 1.0)
    )

    rel_dE_safe = rel_dE + eps

    # Band loss: zero inside [E_lower, E_upper], quadratic outside
    import math
    from src.losses import band_loss_zero_inside_where

    loss_energy = band_loss_zero_inside_where(
        torch.log(rel_dE_safe),
        math.log(config.E_lower),
        math.log(config.E_upper)
    )

    return loss_energy.mean()
```

### Feature Extraction with ModelAdapter (Existing Pattern)
```python
# Source: model_adapter.py lines 38-55, 70-96
# Use this instead of manual feature construction

feats = adapter.build_feature_tensor(
    state=particle,
    history_buffer=history_buffer  # None for analytic mode
)

if feats.dim() == 1:
    feats = feats.unsqueeze(0)  # (F,) -> (1, F)

# Now feats has correct shape for model input
params = model(feats)  # (B, 2)
```

### History Buffer Update (Existing Pattern)
```python
# Source: model_adapter.py lines 97-114
# Always use ModelAdapter to update history (handles deduplication)

adapter.update_history(
    state=accepted_particle,
    history_buffer=history_buffer,
    token=None  # Optional deduplication token
)

# Internally calls history_buffer.push(state)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Multi-step rollout per loss eval | Single-step with accept/reject | Phase 3 (2026-01-20) | Training samples guaranteed to satisfy energy conservation |
| Fixed dt for all training steps | Predicted dt accepted only if passes threshold | Phase 3 (2026-01-20) | Model learns physically valid timesteps |
| Single optimizer step per epoch | Variable optimizer steps until acceptance | Phase 3 (2026-01-20) | Adaptive training effort per state |
| Oldest-state repeat for history padding | Zero-padding for incomplete history | Phase 2 (2026-01-20) | Cleaner bootstrap, warmup steps discarded |

**Deprecated/outdated:**
- Multi-step loss evaluation in Part 1 (still used in Part 2, Phase 4)
- Direct calls to loss_fn_batch in training loop (will be refactored in Phase 5)

## Open Questions

1. **Should retrain loop have a diagnostic max iteration limit?**
   - What we know: Requirements say "no retry limit" (user preference for infinite loop)
   - What's unclear: Whether to add optional `max_retrain_iterations` for debugging mode
   - Recommendation: Add as config parameter with default=None (unlimited), allow users to set for diagnostics

2. **Where should collect_trajectory_step() live initially?**
   - What we know: runner.py has all training logic currently, Phase 6 will refactor to trainer module
   - What's unclear: Create separate trainer_two_phase.py now, or add to runner.py first?
   - Recommendation: Add to runner.py in Phase 3-5, refactor to src/trainer_two_phase.py in Phase 6 integration

3. **Should history buffer be cleared between epochs?**
   - What we know: History buffer holds last K states for features
   - What's unclear: Whether to reset history at epoch boundaries or let it span epochs
   - Recommendation: Don't reset - history should be continuous across epochs for temporal coherence

4. **What to log for Part 1 diagnostics?**
   - What we know: W&B logging exists, tracks loss and energy errors
   - What's unclear: Which Part 1 metrics most useful for debugging (acceptance rate, retrain iterations, etc.)
   - Recommendation: Log per-step: retrain_iterations, final_energy_error; per-epoch: acceptance_rate (should be 1.0 by design), mean_retrain_iterations

## Sources

### Primary (HIGH confidence)
- /u/gkerex/projects/AITimeStepper/run/runner.py - Current training loop (lines 225-395)
- /u/gkerex/projects/AITimeStepper/src/particle.py - ParticleTorch integration and energy (lines 178-428)
- /u/gkerex/projects/AITimeStepper/src/losses.py - Loss functions, band_loss pattern (lines 153-324)
- /u/gkerex/projects/AITimeStepper/src/losses_history.py - History-aware losses (lines 12-247)
- /u/gkerex/projects/AITimeStepper/src/model_adapter.py - Feature extraction abstraction (lines 14-115)
- /u/gkerex/projects/AITimeStepper/src/history_buffer.py - History buffer implementation with zero-padding (lines 26-465)
- /u/gkerex/projects/AITimeStepper/src/config.py - Config dataclass with new parameters (lines 10-282)
- /u/gkerex/projects/AITimeStepper/.planning/REQUIREMENTS.md - TRAIN-01 through TRAIN-04, HIST-02 requirements
- /u/gkerex/projects/AITimeStepper/.planning/ROADMAP.md - Phase 3 success criteria

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Existing PyTorch patterns, no new dependencies
- Architecture: HIGH - Clear accept/reject pattern from requirements, existing primitives well-understood
- Pitfalls: HIGH - Identified from existing loss function patterns and PyTorch memory management

**Research date:** 2026-01-20
**Valid until:** 60 days (stable codebase, internal project)
