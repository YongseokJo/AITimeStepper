# Phase 7: Cleanup Legacy Code - Research

**Researched:** 2026-01-21
**Domain:** Code cleanup / Python module maintenance
**Confidence:** HIGH

## Summary

Phase 7 is a code hygiene phase focused on removing obsolete code now that Phase 6 has successfully integrated the two-phase training system into `run/runner.py`. The research examined the entire codebase to identify:
1. Unused imports in `run/runner.py`
2. Legacy scripts in `run/legacy/` directory
3. Obsolete loss function definitions
4. Obsolete trainer functions
5. Comment/documentation updates needed

The cleanup is straightforward because the new system has clear boundaries. The two-phase training (`run_two_phase_training`) internally uses `band_loss_zero_inside_where` from `src/losses.py` but does NOT use the external-facing loss functions (`loss_fn_batch`, `loss_fn_batch_history`, `loss_fn_batch_history_batch`).

**Primary recommendation:** Remove unused imports from `runner.py`, delete legacy scripts, and add deprecation notices to loss functions that are no longer called from the main training path. Keep the loss functions themselves available for backward compatibility and potential direct usage.

## Standard Stack

### Files to Modify

| File | Action | Why |
|------|--------|-----|
| `run/runner.py` | Remove unused imports | loss_fn_* imports are no longer called |
| `src/losses.py` | Mark as legacy (comments) | Functions still exported but not used in main training |
| `src/losses_history.py` | Mark as legacy (comments) | Functions still exported but not used in main training |
| `src/__init__.py` | Keep as-is (for now) | Maintain backward compatibility |

### Files to Delete

| File | Reason |
|------|--------|
| `run/legacy/ML_history_wandb.py` | Replaced by `run/runner.py train` with two-phase system |
| `run/legacy/ML_history_multi_wandb.py` | Replaced by `run/runner.py train` (multi-orbit deferred with warning) |
| `run/legacy/integration_sanity.py` | One-off test script, functionality covered by `tests/` |
| `run/legacy/tidal_sanity.py` | One-off test script, tidal field tested elsewhere |

### Files to Potentially Move to Obsolete

| File | Current State | Recommendation |
|------|---------------|----------------|
| `src/trainer.py` | Contains old `save_checkpoint` + DeepSpeed functions | Move to `src/obsolete/` - contains duplicate `save_checkpoint` |
| `src/obsolete/losses.py` | Already in obsolete folder | Keep as-is |

## Architecture Patterns

### Current Import Structure in runner.py

```python
from src import (
    Config,
    FullyConnectedNN,
    HistoryBuffer,
    ModelAdapter,
    load_model_state,
    load_config_from_checkpoint,
    loss_fn_batch,              # UNUSED - remove
    loss_fn_batch_history,      # UNUSED - remove
    loss_fn_batch_history_batch, # UNUSED - remove
    make_particle,
    run_two_phase_training,
    save_checkpoint,
    stack_particles,
)
```

### Target Import Structure

```python
from src import (
    Config,
    FullyConnectedNN,
    HistoryBuffer,
    ModelAdapter,
    load_model_state,
    load_config_from_checkpoint,
    make_particle,
    run_two_phase_training,
    save_checkpoint,
    stack_particles,
)
```

### Dependency Analysis

The new two-phase training system has these internal dependencies:

```
run_two_phase_training (unified_training.py)
  |
  +-- train_epoch_two_phase
        |
        +-- collect_trajectory (trajectory_collection.py)
        |     |
        |     +-- collect_trajectory_step
        |     |     |
        |     |     +-- attempt_single_step
        |     |     +-- check_energy_threshold
        |     |     +-- compute_single_step_loss
        |     |           |
        |     |           +-- band_loss_zero_inside_where (losses.py)
        |     |
        |     +-- history_buffer.push()
        |
        +-- generalize_on_trajectory (generalization_training.py)
              |
              +-- evaluate_minibatch
              |     |
              |     +-- compute_single_step_loss (trajectory_collection.py)
              |           |
              |           +-- band_loss_zero_inside_where (losses.py)
              |
              +-- sample_minibatch
```

**Key observation:** The training system uses `band_loss_zero_inside_where` from `losses.py` but does NOT use:
- `loss_fn` (deprecated, has print statements)
- `loss_fn_1` (deprecated)
- `loss_fn_batch` (replaced by internal compute_single_step_loss)
- `loss_fn_batch_history` (replaced)
- `loss_fn_batch_history_batch` (replaced)

### What Must Stay in losses.py

```python
# MUST KEEP - used by trajectory_collection.py and generalization_training.py
def band_loss_zero_inside_where(rel_dE, E_lower, E_upper):
    loss_below = (E_lower - rel_dE).clamp(min=0)**2
    loss_above = (rel_dE - E_upper).clamp(min=0)**2
    return loss_below + loss_above

# MUST KEEP - used by band_loss (module-level constants)
float_info = torch.finfo(torch.float)
double_info = torch.finfo(torch.double)
double_tiny = double_info.tiny
float_tiny = float_info.tiny
eps = float_tiny
```

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Loss computation during cleanup | New loss functions | Keep `band_loss_zero_inside_where` | Already integrated in new system |
| Checkpoint saving | New checkpoint format | Keep existing `save_checkpoint` in checkpoint.py | Works with simulation mode |
| Deleting git-tracked files | Manual rm | `git rm` | Properly tracks file removal |

**Key insight:** This phase is about removal, not replacement. The new training system is already complete; we're just cleaning up unused code.

## Common Pitfalls

### Pitfall 1: Breaking Backward Compatibility
**What goes wrong:** Removing loss_fn_batch from src/__init__.py breaks external code
**Why it happens:** User scripts may import loss_fn_batch directly
**How to avoid:** Keep exports in __init__.py but add deprecation warning at module level:
```python
# In src/losses.py, add docstrings
def loss_fn_batch(...):
    """
    DEPRECATED: This function is no longer used by the main training system.
    The two-phase training system uses internal loss computation via
    `compute_single_step_loss` in `trajectory_collection.py`.

    Kept for backward compatibility. May be removed in a future version.
    ...
    """
```
**Warning signs:** ImportError in user scripts

### Pitfall 2: Removing src/trainer.py save_checkpoint
**What goes wrong:** Some code path still uses trainer.save_checkpoint
**Why it happens:** There are TWO save_checkpoint functions:
  - `src/trainer.py` (old, with duplicated model_state keys)
  - `src/checkpoint.py` (canonical, used by new system)
**How to avoid:** Check all imports of save_checkpoint before removing:
```bash
grep -r "from.*trainer.*import" --include="*.py"
grep -r "trainer\.save_checkpoint" --include="*.py"
```
**Warning signs:** Wrong checkpoint format, simulation mode fails to load

### Pitfall 3: Deleting Legacy Scripts That Have Useful Patterns
**What goes wrong:** User loses reference implementation for custom training
**Why it happens:** Legacy scripts show certain patterns (optuna integration, custom loops)
**How to avoid:**
- Document patterns before deletion in CLAUDE.md or USAGE.md
- Keep one example script if patterns are unique
**Warning signs:** User asks "how do I customize training?" and has no reference

### Pitfall 4: Breaking Tests
**What goes wrong:** Tests import removed functions
**Why it happens:** Test files may use loss_fn_batch directly for unit testing
**How to avoid:** Check test files before removal:
```bash
grep -r "loss_fn_batch" tests/
```
**Current state:** No tests in tests/ directly import loss_fn_batch
**Warning signs:** Test failures after cleanup

### Pitfall 5: Git Status Shows Untracked Legacy Files
**What goes wrong:** Some legacy files may already be git-rm'd but visible in working directory
**Why it happens:** Git status shows "D" for deleted files pending commit
**How to avoid:** Check git status before planning deletions:
```
 D run/ML_history_multi_wandb.py  # Already staged for deletion
 D run/ML_history_wandb.py        # Already staged for deletion
 D run/integration_sanity.py      # Already staged for deletion
 D run/tidal_sanity.py            # Already staged for deletion
```
**Current state:** These files show as deleted in git status but have copies in run/legacy/
**Warning signs:** Double-deleting or confusion about what exists

## Code Examples

### Pattern 1: Adding Deprecation Notice to Functions

```python
# In src/losses.py

def loss_fn_batch(
    model,
    particle: "ParticleTorch",
    n_steps: int = 1,
    ...
):
    """
    Compute physics-informed loss for batched particle system.

    .. deprecated::
        This function is no longer used by the main training pipeline.
        The two-phase training system (run_two_phase_training) uses internal
        loss computation via `trajectory_collection.compute_single_step_loss`.

        Kept for backward compatibility and direct usage scenarios.
        Consider using the two-phase training system instead.

    Args:
        model: Neural network that predicts dt
        ...
    """
    # Function body unchanged
```

### Pattern 2: Removing Unused Imports

```python
# In run/runner.py

# BEFORE:
from src import (
    Config,
    FullyConnectedNN,
    HistoryBuffer,
    ModelAdapter,
    load_model_state,
    load_config_from_checkpoint,
    loss_fn_batch,                  # REMOVE
    loss_fn_batch_history,          # REMOVE
    loss_fn_batch_history_batch,    # REMOVE
    make_particle,
    run_two_phase_training,
    save_checkpoint,
    stack_particles,
)

# AFTER:
from src import (
    Config,
    FullyConnectedNN,
    HistoryBuffer,
    ModelAdapter,
    load_model_state,
    load_config_from_checkpoint,
    make_particle,
    run_two_phase_training,
    save_checkpoint,
    stack_particles,
)
```

### Pattern 3: Git Removal of Legacy Scripts

```bash
# Check what exists
ls -la run/legacy/

# Remove legacy directory and contents
git rm -r run/legacy/

# Commit the removal
git commit -m "chore: remove legacy training scripts

Removed:
- run/legacy/ML_history_wandb.py
- run/legacy/ML_history_multi_wandb.py
- run/legacy/integration_sanity.py
- run/legacy/tidal_sanity.py

These scripts are replaced by:
- run/runner.py train (with two-phase training system)
- tests/test_trajectory_collection.py (integration tests)
"
```

### Pattern 4: Updating src/trainer.py Comments

```python
# In src/trainer.py - add header comment

"""
Legacy trainer utilities.

DEPRECATED: Most functions in this module are no longer used by the main
training pipeline. The canonical training entry point is:

    run/runner.py train

Which uses the two-phase training system from:
    src/unified_training.py
    src/trajectory_collection.py
    src/generalization_training.py

Functions kept for backward compatibility:
- train_one_epoch, validate: Legacy epoch training (DeepSpeed compatible)
- save_checkpoint: DUPLICATE of src/checkpoint.py, prefer checkpoint.py

For new training implementations, see:
- run_two_phase_training()
- train_epoch_two_phase()
- collect_trajectory()
- generalize_on_trajectory()
"""
```

## State of the Art

| Old Approach | New Approach (Post-Phase 6) | Status |
|--------------|----------------------------|--------|
| `run/ML_history_wandb.py` | `run/runner.py train` | REMOVE old |
| `run/ML_history_multi_wandb.py` | `run/runner.py train` (warns if multi-orbit) | REMOVE old |
| `loss_fn_batch` in training | `compute_single_step_loss` (internal) | DEPRECATE, keep for compat |
| `src/trainer.py` save_checkpoint | `src/checkpoint.py` save_checkpoint | DOCUMENT duplicate |

**Fully deprecated (safe to mark as legacy):**
- `loss_fn` - Has print statements, never used in modern code
- `loss_fn_1` - Superseded by loss_fn_batch
- `loss_fn_batch` - Replaced by internal loss in trajectory_collection
- `loss_fn_batch_history` - Replaced by internal loss
- `loss_fn_batch_history_batch` - Replaced (multi-orbit deferred)

**Still actively used:**
- `band_loss_zero_inside_where` - Core loss pattern, used by new system
- `save_checkpoint` (checkpoint.py) - Used by run_two_phase_training

## Open Questions

### Q1: Should we move loss functions to src/obsolete/?
- **What we know:** Functions are no longer used by main training
- **What's unclear:** External users may import them directly
- **Recommendation:** Keep in src/losses.py with deprecation notices. Moving would break imports. Consider removal in a future major version.

### Q2: Should we remove src/trainer.py entirely?
- **What we know:** Contains duplicate save_checkpoint and DeepSpeed functions
- **What's unclear:** DeepSpeed functions may be wanted for distributed training
- **Recommendation:** Keep but add deprecation header. The DeepSpeed integration might be valuable for future work.

### Q3: Should we update src/__init__.py to warn on deprecated imports?
- **What we know:** Could add warnings like `warnings.warn("loss_fn_batch deprecated...")`
- **What's unclear:** Would create noise for users who want to use these functions intentionally
- **Recommendation:** No runtime warnings. Use docstring deprecation only. Users who grep for the function will see the notice.

## Cleanup Checklist

### runner.py Cleanup
- [ ] Remove `loss_fn_batch` from imports
- [ ] Remove `loss_fn_batch_history` from imports
- [ ] Remove `loss_fn_batch_history_batch` from imports
- [ ] Verify imports still work (run `python -c "from run.runner import main"`)

### Legacy Scripts Cleanup
- [ ] Delete `run/legacy/ML_history_wandb.py`
- [ ] Delete `run/legacy/ML_history_multi_wandb.py`
- [ ] Delete `run/legacy/integration_sanity.py`
- [ ] Delete `run/legacy/tidal_sanity.py`
- [ ] Remove `run/legacy/` directory

### Documentation Updates
- [ ] Add deprecation docstring to `loss_fn` in losses.py
- [ ] Add deprecation docstring to `loss_fn_1` in losses.py
- [ ] Add deprecation docstring to `loss_fn_batch` in losses.py
- [ ] Add deprecation docstring to `loss_fn_batch_history` in losses_history.py
- [ ] Add deprecation docstring to `loss_fn_batch_history_batch` in losses_history.py
- [ ] Add deprecation header to `src/trainer.py`

### Verification
- [ ] Run tests: `pytest tests/`
- [ ] Run basic training: `python run/runner.py train --epochs 3 --num-particles 3`
- [ ] Run basic simulation: `python run/runner.py simulate --num-particles 3 --steps 100`

## Sources

### Primary (HIGH confidence)
- `/u/gkerex/projects/AITimeStepper/run/runner.py` - Current imports verified
- `/u/gkerex/projects/AITimeStepper/src/__init__.py` - Export structure verified
- `/u/gkerex/projects/AITimeStepper/src/losses.py` - Function definitions analyzed
- `/u/gkerex/projects/AITimeStepper/src/losses_history.py` - Function definitions analyzed
- `/u/gkerex/projects/AITimeStepper/src/trajectory_collection.py` - New loss usage verified
- `/u/gkerex/projects/AITimeStepper/src/generalization_training.py` - New loss usage verified
- `/u/gkerex/projects/AITimeStepper/run/legacy/` - Legacy scripts examined

### Secondary (MEDIUM confidence)
- `/u/gkerex/projects/AITimeStepper/.planning/phases/06-integration-into-runner/06-RESEARCH.md` - Phase 6 recommendations
- Git status output - File state verified

## Metadata

**Confidence breakdown:**
- Import cleanup: HIGH - Direct code analysis
- Legacy script removal: HIGH - Files examined, replacements verified
- Deprecation strategy: MEDIUM - Backward compatibility is a judgment call

**Research date:** 2026-01-21
**Valid until:** 2026-02-21 (30 days - internal codebase, stable domain)
