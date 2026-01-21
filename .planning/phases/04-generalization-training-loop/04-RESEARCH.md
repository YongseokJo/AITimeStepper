# Phase 4: Part 2 - Generalization Training Loop - Research

**Researched:** 2026-01-21
**Domain:** PyTorch minibatch training with convergence criterion
**Confidence:** HIGH

## Summary

Phase 4 implements the "Part 2" of the two-phase training routine: generalization training on collected trajectory. The trajectory collected in Part 1 (Phase 3) contains validated `(ParticleTorch, dt)` tuples that passed energy threshold. Part 2 samples random minibatches from this trajectory and trains until ALL samples in each minibatch pass the energy threshold.

The implementation follows established PyTorch patterns for minibatch training, using `torch.randperm()` for random index sampling and a convergence-based while loop instead of fixed epochs. The key insight from the existing codebase (legacy `ML_history_wandb.py`) is the `rel_dE_full.all()` convergence check pattern.

**Primary recommendation:** Use `torch.randperm()` for index sampling with a simple convergence loop, reusing the existing `attempt_single_step()` and `check_energy_threshold()` primitives from Phase 3.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.x | Tensor operations and autograd | Already used in project |
| `torch.randperm()` | built-in | Random index sampling | Official PyTorch API for shuffling |
| `random.sample()` | Python stdlib | Alternative sampling | Used in legacy code |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `dataclasses` | stdlib | Structured return types | For metrics/results |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `torch.randperm()` | `random.sample()` | `random.sample()` works directly on lists, but `torch.randperm()` is GPU-compatible and faster for large datasets |
| Manual loop | `torch.utils.data.DataLoader` | DataLoader is overkill for simple list sampling; trajectory is small (<100 items typically) |
| `torchrl.data.ReplayBuffer` | N/A | Too heavyweight; trajectory is simple list, not complex RL experience tuples |

**Installation:**
No additional dependencies required - all functionality exists in base PyTorch.

## Architecture Patterns

### Recommended Project Structure
```
src/
├── trajectory_collection.py  # Phase 3 (already exists)
├── generalization_training.py  # NEW: Phase 4 module
└── config.py                   # Configuration (already exists)
```

### Pattern 1: Random Minibatch Sampling
**What:** Sample random indices from trajectory using `torch.randperm()` or `random.sample()`
**When to use:** When trajectory is a Python list of tuples
**Example:**
```python
# Source: PyTorch docs (torch.randperm)
# For list-based trajectory, random.sample is simpler
import random

def sample_minibatch(trajectory, batch_size):
    """Sample random minibatch from trajectory."""
    batch_size = min(batch_size, len(trajectory))
    return random.sample(trajectory, k=batch_size)
```

### Pattern 2: Convergence-Based Training Loop
**What:** Train until convergence criterion is met, not for fixed iterations
**When to use:** When requirement is "train until all samples pass"
**Example:**
```python
# Source: Legacy ML_history_wandb.py convergence pattern
max_iterations = config.replay_steps  # safety limit

for iteration in range(max_iterations):
    minibatch = sample_minibatch(trajectory, batch_size)

    # Forward pass on all samples
    all_pass, losses, metrics = evaluate_minibatch(minibatch)

    if all_pass:
        # Convergence achieved
        return True, iteration + 1

    # Backprop and update
    loss = aggregate_loss(losses)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Did not converge within max_iterations
return False, max_iterations
```

### Pattern 3: Single-Step Prediction per Sample
**What:** Each trajectory sample gets single-step dt prediction and integration
**When to use:** TRAIN-07 requirement - single timestep predictions
**Example:**
```python
# Source: trajectory_collection.py existing pattern
def evaluate_single_sample(model, particle, config, adapter, history_buffer):
    """Evaluate one sample with single-step prediction."""
    # Reuse Phase 3 primitive
    p_new, dt, E0, E1 = attempt_single_step(
        model, particle, config, adapter, history_buffer
    )
    passed, rel_dE = check_energy_threshold(E0, E1, config.energy_threshold)
    return passed, rel_dE, E0, E1
```

### Pattern 4: Detached Samples for Replay
**What:** Trajectory samples are already detached from computation graph
**When to use:** When replaying from stored trajectory
**Example:**
```python
# Source: trajectory_collection.py collect_trajectory() line 375
# Samples in trajectory are already clone_detached()
for particle, dt in trajectory:
    # particle is detached - safe to use for new forward pass
    # Creates fresh computation graph
    p_new, dt_pred, E0, E1 = attempt_single_step(model, particle, ...)
```

### Anti-Patterns to Avoid
- **Fixed iteration count when convergence required:** TRAIN-06 requires training until ALL samples pass, not for N iterations. Use max iterations as safety only.
- **Batching incompatible particle states:** Trajectory samples may have different positions/velocities. Process individually or use `stack_particles()` carefully.
- **Graph accumulation across iterations:** Each iteration should start with fresh clone_detached() particles.
- **Modifying trajectory samples:** Trajectory is read-only reference; never modify stored particles.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Single-step prediction | Custom forward pass | `attempt_single_step()` from Phase 3 | Already handles clone_detached, energy computation |
| Energy threshold check | Custom comparison | `check_energy_threshold()` from Phase 3 | Handles batched/scalar, relative error |
| Loss computation | Custom loss | `compute_single_step_loss()` from Phase 3 | Band loss with proper guards |
| Feature construction | Direct model call | `ModelAdapter.build_feature_tensor()` | Handles analytic vs history modes |

**Key insight:** Phase 3 primitives are designed for reuse. Part 2 generalization is essentially "call Phase 3 primitives on trajectory samples until all pass."

## Common Pitfalls

### Pitfall 1: Infinite Loop Without Safety Limit
**What goes wrong:** Training forever if model cannot converge
**Why it happens:** "Train until all pass" has no natural termination for bad models
**How to avoid:** Use `config.replay_steps` as max iteration limit (default: 1000)
**Warning signs:** Inner loop running for 10,000+ iterations

### Pitfall 2: History Buffer State Inconsistency
**What goes wrong:** Using current history buffer state for old trajectory samples
**Why it happens:** History buffer was updated during Part 1 collection
**How to avoid:** Two options:
1. Store history snapshot with each trajectory sample (like legacy code)
2. Don't use history features in Part 2 (trajectory samples are already validated)
**Recommendation:** Option 2 is simpler - recompute analytic features only, or store history snapshots if history is critical

### Pitfall 3: Batch Size Larger Than Trajectory
**What goes wrong:** Attempting to sample more items than available
**Why it happens:** Config may have large batch_size, trajectory may be small
**How to avoid:** `batch_size = min(config.replay_batch_size, len(trajectory))`
**Warning signs:** IndexError or crash during sampling

### Pitfall 4: All Samples Already Pass
**What goes wrong:** Part 2 loop does nothing because trajectory is perfect
**Why it happens:** Part 1 already ensured all samples pass threshold
**How to avoid:** This is expected behavior - return immediately as "converged"
**Note:** This is not really a pitfall, but a valid edge case

### Pitfall 5: Empty Trajectory
**What goes wrong:** No samples to train on
**Why it happens:** Steps_per_epoch <= history_len (all warmup, no trajectory)
**How to avoid:** Check `if not trajectory: return early` with appropriate status
**Warning signs:** Configuration warning from Phase 3 `collect_trajectory()`

## Code Examples

Verified patterns from official sources and existing codebase:

### Main Generalization Function Signature
```python
# Based on requirements TRAIN-05, TRAIN-06, TRAIN-07
def generalize_on_trajectory(
    model: torch.nn.Module,
    trajectory: List[Tuple[ParticleTorch, float]],
    optimizer: torch.optim.Optimizer,
    config: Config,
    adapter: ModelAdapter,
) -> Tuple[bool, int, Dict[str, Any]]:
    """
    Train on random minibatches from trajectory until all pass.

    Args:
        model: Neural network for dt prediction
        trajectory: List of (particle, dt) from collect_trajectory()
        optimizer: PyTorch optimizer
        config: Config with replay_batch_size, replay_steps, energy_threshold
        adapter: ModelAdapter for feature construction

    Returns:
        (converged, iteration_count, metrics) where:
        - converged: True if all samples passed threshold
        - iteration_count: Number of training iterations
        - metrics: Dict with 'mean_energy_error', 'max_energy_error', etc.
    """
```

### Minibatch Evaluation Pattern
```python
# Source: Pattern from legacy ML_history_wandb.py lines 339-348
def check_all_pass(rel_dE_list: List[torch.Tensor], threshold: float) -> bool:
    """Check if all samples pass energy threshold."""
    if not rel_dE_list:
        return True  # Empty trajectory trivially passes

    all_pass = all(rel_dE.item() < threshold for rel_dE in rel_dE_list)
    return all_pass
```

### Loss Aggregation Pattern
```python
# Aggregate losses from minibatch samples
def aggregate_minibatch_loss(
    losses: List[torch.Tensor]
) -> torch.Tensor:
    """Combine per-sample losses into single scalar."""
    if not losses:
        return torch.tensor(0.0, requires_grad=True)
    return torch.stack(losses).mean()
```

### torch.randperm Usage
```python
# Source: PyTorch docs https://docs.pytorch.org/docs/stable/generated/torch.randperm.html
import torch

# Generate random indices for minibatch
n_samples = len(trajectory)
batch_size = min(config.replay_batch_size, n_samples)
indices = torch.randperm(n_samples)[:batch_size]

# Use indices to sample
minibatch = [trajectory[i] for i in indices.tolist()]
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Fixed epochs | Convergence-based | Design decision | Ensures all samples pass |
| Multi-step loss | Single-step loss | TRAIN-07 | Simpler, more targeted training |
| Full batch | Minibatch sampling | TRAIN-05 | Better generalization, faster |

**Deprecated/outdated:**
- Multi-step integration per sample: Phase 3 established single-step pattern
- Custom replay buffer class: Simple list is sufficient for trajectory

## Implementation Strategy

### Recommended Approach

1. **Reuse Phase 3 primitives** - `attempt_single_step()`, `check_energy_threshold()`, `compute_single_step_loss()`

2. **Simple minibatch loop:**
```python
for iteration in range(max_iterations):
    minibatch = random.sample(trajectory, batch_size)

    losses = []
    rel_dE_list = []

    for particle, _ in minibatch:
        p_new, dt, E0, E1 = attempt_single_step(model, particle.clone_detached(), config, adapter)
        passed, rel_dE = check_energy_threshold(E0, E1, config.energy_threshold)

        if not passed:
            loss = compute_single_step_loss(E0, E1, config)
            losses.append(loss)

        rel_dE_list.append(rel_dE)

    # Check convergence (all pass)
    if not losses:  # All samples passed
        return True, iteration + 1, metrics

    # Train on failed samples
    total_loss = torch.stack(losses).mean()
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

return False, max_iterations, metrics
```

3. **Config parameters to use:**
- `config.replay_batch_size` (default 512) - minibatch size
- `config.replay_steps` (default 1000) - max iterations
- `config.energy_threshold` (default 2e-4) - convergence criterion

### Alternative: Batched Approach

If performance is critical, can batch all samples:

```python
# Stack all particles for batch processing
particles = [p for p, _ in minibatch]
batch_state = stack_particles(particles)

# Batch forward pass
# ... more complex, requires history handling
```

Trade-off: More complex, but faster for large minibatches. Recommendation: Start with simple loop, optimize later if needed.

## Open Questions

Things that couldn't be fully resolved:

1. **History buffer handling in Part 2**
   - What we know: Phase 3 trajectory samples don't store history snapshots
   - What's unclear: Should Part 2 use history features, or analytic-only?
   - Recommendation: Use analytic features for simplicity. Trajectory samples were already validated with their history context. Part 2 is about generalizing the model, not exact reconstruction.

2. **Batch size vs trajectory size**
   - What we know: Config has `replay_batch_size=512`, but trajectory may be small
   - What's unclear: Should batch cover entire trajectory, or subsample?
   - Recommendation: `min(config.replay_batch_size, len(trajectory))` - sample at most available

3. **Convergence without training**
   - What we know: If trajectory passes immediately, return success
   - What's unclear: Is this a valid outcome or indicates Phase 1 threshold was too loose?
   - Recommendation: Valid outcome - model generalized well from Part 1 training

## Sources

### Primary (HIGH confidence)
- `/u/gkerex/projects/AITimeStepper/src/trajectory_collection.py` - Phase 3 primitives
- `/u/gkerex/projects/AITimeStepper/run/legacy/ML_history_wandb.py` lines 248-373 - Replay buffer training pattern
- `/u/gkerex/projects/AITimeStepper/src/config.py` - Config parameters
- [PyTorch torch.randperm](https://docs.pytorch.org/docs/stable/generated/torch.randperm.html) - Random permutation API

### Secondary (MEDIUM confidence)
- [PyTorch Training Tutorial](https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html) - General training loop patterns
- [TorchRL Replay Buffers](https://docs.pytorch.org/rl/main/tutorials/rb_tutorial.html) - Replay buffer concepts (not used directly)

### Tertiary (LOW confidence)
- Web search results on convergence-based training - General patterns, not PyTorch-specific

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Using existing PyTorch APIs and codebase patterns
- Architecture: HIGH - Clear requirements, existing Phase 3 patterns to follow
- Pitfalls: MEDIUM - Based on codebase analysis and general PyTorch experience

**Research date:** 2026-01-21
**Valid until:** 30 days (stable domain, no external dependencies changing)

## Requirements Mapping

| Requirement | Implementation Strategy |
|-------------|------------------------|
| TRAIN-05 | `random.sample()` or `torch.randperm()` from trajectory list |
| TRAIN-06 | Convergence loop with `all(rel_dE < threshold)` check |
| TRAIN-07 | Reuse `attempt_single_step()` for single timestep predictions |
