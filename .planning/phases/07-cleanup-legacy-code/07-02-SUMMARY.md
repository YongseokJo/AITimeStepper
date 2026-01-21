# Phase 07 Plan 02 Summary: Add Deprecation Notices to Legacy Code

## Completed: 2026-01-21

## Objective
Add deprecation notices to legacy loss functions and trainer module to document which functions are no longer used by the main training pipeline, guiding future developers and external users.

## Changes Made

### 1. src/losses.py
Added deprecation docstrings to three functions:

**loss_fn**
- Status: DEPRECATED
- Reason: Early experimental function with print statements, never used in modern code
- Replacement: Superseded by loss_fn_batch and trajectory-based training

**loss_fn_1**
- Status: DEPRECATED
- Reason: Single-system (non-batched) loss function
- Replacement: Superseded by loss_fn_batch, which is itself deprecated in favor of trajectory_collection.py

**loss_fn_batch**
- Status: DEPRECATED
- Reason: Primary loss function for original training loop
- Replacement: run_two_phase_training() from trajectory_collection.py with internal compute_single_step_loss()

**band_loss_zero_inside_where**
- Status: NOT deprecated - actively used by new system
- Usage: Imported by src/trajectory_collection.py and src/losses_history.py

### 2. src/losses_history.py
Added deprecation docstrings to two functions:

**loss_fn_batch_history**
- Status: DEPRECATED
- Reason: History-aware loss with explicit HistoryBuffer management
- Replacement: trajectory_collection.compute_single_step_loss() handles history internally

**loss_fn_batch_history_batch**
- Status: DEPRECATED
- Reason: Multi-orbit batched history-aware loss
- Replacement: run_two_phase_training() handles multi-orbit with warning and single-orbit fallback

### 3. src/trainer.py
Added module-level deprecation docstring explaining:
- Contains legacy training functions from earlier codebase version
- train_one_epoch(), validate() etc. designed for data-loader-based training
- Not used in physics-informed trajectory training approach
- save_checkpoint() is a duplicate of src/checkpoint.py (prefer checkpoint.py)

## Verification

All imports verified working:
```
from src.losses import loss_fn_batch, band_loss_zero_inside_where  # OK
from src.losses_history import loss_fn_batch_history, loss_fn_batch_history_batch  # OK
from src.trainer import save_checkpoint  # OK
```

band_loss_zero_inside_where verified functional:
```python
x = torch.tensor([0.5, 1.5, 2.5])
band_loss_zero_inside_where(x, 1.0, 2.0)  # tensor([0.2500, 0.0000, 0.2500])
```

## Files Modified
- `/u/gkerex/projects/AITimeStepper/src/losses.py`
- `/u/gkerex/projects/AITimeStepper/src/losses_history.py`
- `/u/gkerex/projects/AITimeStepper/src/trainer.py`

## Success Criteria Met
1. [x] loss_fn, loss_fn_1, loss_fn_batch in src/losses.py have deprecation docstrings
2. [x] loss_fn_batch_history, loss_fn_batch_history_batch in src/losses_history.py have deprecation docstrings
3. [x] src/trainer.py has module-level deprecation header
4. [x] All imports still work (backward compatibility preserved)
5. [x] band_loss_zero_inside_where is NOT deprecated (actively used by trajectory_collection.py)

## Key Links Verified
- trajectory_collection.py imports band_loss_zero_inside_where from losses.py (line 22)
- losses_history.py imports band_loss_zero_inside_where from losses.py (line 9)
