# Plan 04-01 Execution Summary

**Phase:** 04-generalization-training-loop
**Plan:** 04-01
**Date:** 2026-01-21
**Status:** Complete

## Objective

Implement the generalization training loop (Part 2) that trains on random minibatches from collected trajectory until all samples pass the energy threshold.

## Tasks Completed

### Task 1: Create generalization_training.py with minibatch sampling
- **Files Modified:** `src/generalization_training.py` (new)
- **Commit:** d84477b

Created `src/generalization_training.py` module (241 lines) with complete implementation:

1. **`sample_minibatch()`** - Random minibatch sampling
   - Uses `random.sample()` for uniform sampling without replacement
   - Caps batch_size at trajectory length with `min()`
   - Returns list of (particle, dt) tuples

2. **`evaluate_minibatch()`** - Single-step evaluation of minibatch
   - Loops over each (particle, dt) in minibatch
   - Calls `attempt_single_step()` for each sample
   - Calls `check_energy_threshold()` to validate energy conservation
   - Computes loss with `compute_single_step_loss()` for failed samples
   - Returns (all_pass, losses, rel_dE_list, metrics)

3. **`generalize_on_trajectory()`** - Main convergence loop
   - Handles empty trajectory edge case (returns success immediately)
   - Handles trajectory below min_replay_size (returns success with 'skipped' flag)
   - Main loop: samples minibatch, evaluates, aggregates losses, backprops
   - Convergence criterion: all samples in minibatch pass energy threshold
   - Safety limit: config.replay_steps (default 1000)
   - Returns (converged, iterations, final_metrics)

### Task 2: Export functions from src/__init__.py
- **Files Modified:** `src/__init__.py`
- **Commit:** 80b86c0

Added import block for generalization training exports:
- `generalize_on_trajectory`
- `sample_minibatch`
- `evaluate_minibatch`

### Task 3: Add typing and edge case handling
- **Status:** Completed in Task 1 (combined implementation)
- **Verification:**
  - Complete type annotations for all functions
  - Edge case: empty trajectory returns (True, 0, {metrics})
  - Edge case: trajectory < min_replay_size returns with 'skipped': True
  - Explicit `.item()` conversion for metrics

## Verification Results

- **Syntax Check:** `python3 -m py_compile` passes
- **Line Count:** 241 lines (minimum 120 required)
- **Exports:** All three functions importable from `src`
- **Signature Check:** `generalize_on_trajectory` has all required parameters
- **Key Patterns:**
  - `random.sample()` for minibatch selection (not torch.randperm)
  - Particle immutability preserved via clone_detached in attempt_single_step
  - config.replay_batch_size and config.replay_steps usage
  - Proper loss aggregation with torch.stack().mean()

## Must-Haves Verification

### Truths
- Minibatch samples random indices from trajectory list
- Each sample gets single-step dt prediction and integration
- Convergence loop trains until all samples pass energy threshold
- Max iteration safety limit prevents infinite loops
- Trajectory particles are never modified during replay (immutability)

### Artifacts
- `src/generalization_training.py` exists
- Provides: Generalization training functions
- Exports: `generalize_on_trajectory`, `sample_minibatch`, `evaluate_minibatch`
- 241 lines (min 120)

### Key Links
- `generalization_training.py` -> `trajectory_collection.py` via `from .trajectory_collection import attempt_single_step, check_energy_threshold, compute_single_step_loss`
- `generalization_training.py` -> `config.py` via `config.replay_batch_size`, `config.replay_steps`, `config.energy_threshold`, `config.min_replay_size`

## Implementation Notes

### Design Decisions

1. **Random Sampling with random.sample():** Trajectory is a Python list, so random.sample() is more appropriate than torch.randperm(). This provides uniform sampling without replacement.

2. **Edge Case Handling:** Empty and small trajectories return early with success status. Small trajectories (< min_replay_size) are flagged with 'skipped': True in metrics for debugging.

3. **Loss Aggregation:** Failed samples contribute to a stacked loss tensor, which is then averaged. This ensures gradient scaling is independent of batch size.

4. **Immutability Preservation:** Trajectory particles are never modified because attempt_single_step() creates clone_detached() internally. This is critical for replaying the same trajectory multiple times.

### Pattern Consistency

The implementation follows existing codebase patterns:
- Import of Phase 3 primitives from trajectory_collection.py
- Config parameter usage (replay_batch_size, replay_steps, min_replay_size)
- Metrics dict structure with mean_rel_dE, max_rel_dE, etc.
- Type hints consistent with trajectory_collection.py

## Files Modified

- `src/generalization_training.py` (new file, 241 lines)
- `src/__init__.py` (5 lines added)

## Commits

1. `d84477b` - feat(04-01): create generalization_training.py with minibatch sampling
2. `80b86c0` - feat(04-01): export generalization training functions from src/__init__.py

## Next Steps

Ready for Plan 04-02: Add unit tests for generalization training functions.
