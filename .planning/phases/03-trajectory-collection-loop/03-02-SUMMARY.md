# Plan 03-02 Execution Summary

**Phase:** 03-trajectory-collection-loop
**Plan:** 03-02
**Date:** 2026-01-20
**Status:** Complete

## Objective

Implement the accept/reject retrain loop that collects a single validated trajectory step.

## Tasks Completed

### Task 1: Implement collect_trajectory_step function

- **Files Modified:** `src/trajectory_collection.py`, `src/__init__.py`
- **Commit:** 50a2f42

Implemented `collect_trajectory_step()` function with the following features:

1. **Accept/Reject Loop Structure**
   - Uses `while True` loop with no max iteration limit (by design)
   - Calls `attempt_single_step()` to predict dt and integrate
   - Calls `check_energy_threshold()` to verify energy conservation
   - On acceptance: returns particle, dt, and metrics
   - On rejection: computes loss and retrains

2. **Retrain Mechanism**
   - Computes loss via `compute_single_step_loss()`
   - Executes optimizer cycle: `zero_grad()`, `backward()`, `step()`
   - Fresh clone created at each attempt (inside `attempt_single_step`)

3. **Return Structure**
   - `accepted_particle`: ParticleTorch that passed energy check
   - `accepted_dt`: float value of timestep
   - `metrics`: dict with `retrain_iterations`, `reject_count`, `final_energy_error`

### Task 2: Add diagnostic logging option

- **Parameter:** `max_retrain_warn: int = 1000`
- **Behavior:** Issues `RuntimeWarning` every N iterations
- **Warning Message:** Includes iteration count, current rel_dE, and threshold

This allows debugging stuck retrain loops without breaking the infinite loop requirement.

### Task 3: Export function and update __init__.py

- **File:** `src/__init__.py`
- **Export:** `collect_trajectory_step` added to multi-line import

## Verification Results

All verification commands from the plan passed:

```
$ python -c "from src.trajectory_collection import collect_trajectory_step; print('collect_trajectory_step OK')"
collect_trajectory_step OK

$ python -c "from src.trajectory_collection import collect_trajectory_step; import inspect; sig = inspect.signature(collect_trajectory_step); print('max_retrain_warn' in sig.parameters)"
True

$ python -c "from src import collect_trajectory_step; print('Export OK')"
Export OK

$ python -c "from src import collect_trajectory_step; import inspect; print(inspect.signature(collect_trajectory_step))"
(model: torch.nn.modules.module.Module, particle: src.particle.ParticleTorch, optimizer: torch.optim.optimizer.Optimizer, config: src.config.Config, adapter: src.model_adapter.ModelAdapter, history_buffer: Optional[src.history_buffer.HistoryBuffer] = None, max_retrain_warn: int = 1000) -> Tuple[src.particle.ParticleTorch, float, Dict[str, Any]]
```

## Must-Haves Verification

### Truths
- Retrain loop continues until energy threshold satisfied (no max iteration limit)
- Each iteration performs `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`
- Fresh clone created at start of each attempt (inside `attempt_single_step`)
- Returns accepted particle state and metrics

### Artifacts
- `src/trajectory_collection.py` - provides `collect_trajectory_step` function
- Exports: `collect_trajectory_step`
- Min lines: 100+ (function spans lines 199-278, 80 lines; total file 393 lines)

### Key Links
- `collect_trajectory_step` -> `attempt_single_step` via function call in loop (line 245)
- `collect_trajectory_step` -> `torch.optim` via `optimizer.step()` (line 268)

## Implementation Notes

### Function Signature

```python
def collect_trajectory_step(
    model: torch.nn.Module,
    particle: ParticleTorch,
    optimizer: torch.optim.Optimizer,
    config: Config,
    adapter: ModelAdapter,
    history_buffer: Optional[HistoryBuffer] = None,
    max_retrain_warn: int = 1000,
) -> Tuple[ParticleTorch, float, Dict[str, Any]]:
```

### Key Design Decisions

1. **No Max Iteration Limit:** Per user requirement, the loop runs until energy threshold is satisfied. This design choice prioritizes correctness over bounded runtime.

2. **Diagnostic Warning:** The `max_retrain_warn` parameter provides visibility into long-running loops without interrupting execution. Default of 1000 balances visibility with noise.

3. **Metrics Redundancy:** `reject_count` equals `retrain_iterations` - both are included for clarity in different usage contexts.

4. **Float Return for dt:** The `dt` is extracted as a Python float rather than tensor to simplify downstream usage and storage.

### Requirements Implemented

- **TRAIN-01:** Predict dt, integrate one step, check energy
- **TRAIN-02:** If energy exceeds threshold, reject and retrain on same state
- **TRAIN-03:** Loop until energy within threshold

## Files Modified

- `src/trajectory_collection.py` (80 lines added for `collect_trajectory_step`)
- `src/__init__.py` (export added)

## Next Steps

Ready for Plan 03-03: Implement `collect_trajectory()` that orchestrates N steps per epoch with warmup phase handling (TRAIN-04, HIST-02).
