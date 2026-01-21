# Plan 03-03 Execution Summary

**Phase:** 03-trajectory-collection-loop
**Plan:** 03-03
**Date:** 2026-01-20
**Status:** Complete

## Objective

Implement the trajectory collection orchestrator that collects N validated steps per epoch with warmup phase handling.

## Tasks Completed

### Task 1: Implement collect_trajectory function

- **Files Modified:** `src/trajectory_collection.py`, `src/__init__.py`
- **Commit:** 50a2f42 (combined with Plan 03-02)

Implemented `collect_trajectory()` function with the following features:

1. **N Steps Per Epoch (TRAIN-04)**
   - Loops for `config.steps_per_epoch` iterations
   - Calls `collect_trajectory_step()` to get each accepted step
   - Tracks all step metrics for aggregation

2. **Warmup Phase Handling (HIST-02)**
   - Determines warmup length from `history_len` (when history buffer is enabled)
   - Always pushes to history buffer (even during warmup)
   - Discards first `history_len` steps from returned trajectory
   - Warmup steps fill buffer but don't contribute to loss computation

3. **Return Structure**
   - `trajectory`: List of (ParticleTorch, dt) tuples (warmup excluded)
   - `epoch_metrics`: dict with aggregated statistics:
     - `total_steps`: Number of steps attempted
     - `warmup_discarded`: Number of warmup steps discarded
     - `trajectory_length`: Actual trajectory size
     - `mean_retrain_iterations`: Average retrains per step
     - `mean_energy_error`: Average energy error
     - `max_retrain_iterations`: Maximum retrains in any step

### Task 2: Handle edge cases

1. **Validation:** Raises `ValueError` if `steps_per_epoch < 1`
2. **Warmup Warning:** Issues `UserWarning` when `steps_per_epoch <= history_len` (all steps would be warmup)
3. **Analytic Mode:** When `history_buffer is None`, `warmup_len = 0` (no warmup needed)

### Task 3: Export function and update __init__.py

- **File:** `src/__init__.py`
- **Export:** `collect_trajectory` added to multi-line import

## Verification Results

All verification commands from the plan passed:

```
$ python -c "from src.trajectory_collection import collect_trajectory; print('collect_trajectory OK')"
collect_trajectory OK

$ python -c "from src import collect_trajectory; print('Export OK')"
Export OK

$ python -c "from src import collect_trajectory; import inspect; print(inspect.signature(collect_trajectory))"
(model: torch.nn.Module, particle: ParticleTorch, optimizer: Optimizer, config: Config, adapter: ModelAdapter, history_buffer: Optional[HistoryBuffer] = None) -> Tuple[List[Tuple[ParticleTorch, float]], Dict[str, Any]]
```

## Must-Haves Verification

### Truths
- Collects N steps per epoch using `config.steps_per_epoch`
- Warmup steps (first `history_len`) discarded from trajectory list
- History buffer always updated (even during warmup)
- Returns list of accepted (state, dt) tuples

### Artifacts
- `src/trajectory_collection.py` - provides `collect_trajectory` function
- Exports: `collect_trajectory`
- Min lines: 150+ (total file 393 lines)

### Key Links
- `collect_trajectory` -> `collect_trajectory_step` via function call in loop (line 350)
- `collect_trajectory` -> `history_buffer.push()` for buffer updates (line 363)

## Implementation Notes

### Function Signature

```python
def collect_trajectory(
    model: torch.nn.Module,
    particle: ParticleTorch,
    optimizer: torch.optim.Optimizer,
    config: Config,
    adapter: ModelAdapter,
    history_buffer: Optional[HistoryBuffer] = None,
) -> Tuple[List[Tuple[ParticleTorch, float]], Dict[str, Any]]:
```

### Key Design Decisions

1. **Warmup Calculation:** `warmup_len = config.history_len if (history_buffer is not None and config.history_len > 0) else 0`
   - Only applies warmup when history-aware mode is active
   - Analytic mode (no history buffer) has zero warmup

2. **Buffer Push Before Warmup Check:** History buffer is updated BEFORE checking if the step is warmup. This ensures:
   - Buffer is always populated with latest state
   - Next step always has access to updated features
   - Warmup check only affects trajectory list, not buffer state

3. **Clone Detached Storage:** All particles stored in trajectory use `clone_detached()` to prevent computation graph accumulation.

4. **Aggregated Metrics:** Metrics are computed over ALL steps (including warmup) to provide complete training visibility.

### Requirements Implemented

- **TRAIN-04:** N steps per epoch using `steps_per_epoch` config parameter
- **HIST-02:** Warmup steps discarded from returned trajectory list

## Files Modified

- `src/trajectory_collection.py` (113 lines added for `collect_trajectory`)
- `src/__init__.py` (export added)

## Success Criteria Met

1. `collect_trajectory` exists with correct signature
2. Loops for `config.steps_per_epoch` iterations
3. Always pushes to history buffer (every step)
4. Discards first `history_len` steps from trajectory list
5. Returns (trajectory_list, epoch_metrics)
6. Handles edge case: steps_per_epoch <= history_len
7. Handles analytic mode (no history buffer)

## Next Steps

Ready for Plan 03-04: Integrate trajectory collection with replay buffer for Phase 2 training.
