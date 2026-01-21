# AITimeStepper: Technical Concerns & Debt Analysis

## Executive Summary

This document outlines technical debt, known issues, and fragile areas in the AITimeStepper codebase. The project is in active development with solid foundational architecture but shows signs of rapid iteration that have introduced several maintenance and robustness concerns.

---

## Critical Issues

### 1. **Incomplete Return Value Handling in `losses.py`**
- **Location**: `/u/gkerex/projects/AITimeStepper/src/losses.py`, line 309
- **Issue**: The `loss_fn_batch()` function has an inconsistent return pattern:
  ```python
  if return_particle:
      return loss, metrics, p  # p now represents batched advanced particle(s)
  else:
      return loss, metrics, _  # Returns undefined variable '_'
  ```
- **Severity**: CRITICAL - This will cause `NameError` when `return_particle=False`
- **Impact**: Training fails whenever the return_particle flag is False in batch loss calculations
- **Fix Required**: Should be `return loss, metrics` instead of `return loss, metrics, _`

### 2. **Uninitialized Variable in `losses_history.py`**
- **Location**: `/u/gkerex/projects/AITimeStepper/src/losses_history.py`, line 246
- **Issue**: Same issue as above - returns undefined `_` variable
  ```python
  else:
      return loss, metrics, _  # NameError: _ is not defined
  ```
- **Severity**: CRITICAL
- **Impact**: History-aware loss calculations fail when `return_particle=False`
- **Fix Required**: Same fix as above

---

## High Priority Issues

### 3. **Debug Print Statements in Production Code**
- **Location**: Multiple files
  - `src/losses.py`: Lines 18-20, 27, 76, 94, 100, 107, 126
  - `src/particle.py`: Multiple debug prints
- **Issue**: Leftover debug `print()` statements that clutter output and indicate incomplete development
  ```python
  print("params: ", params, "params shape: ", params.shape)
  print("E0: ", E0, "E1: ", E1, "dt: ", dt, "E: ", E)
  ```
- **Severity**: HIGH
- **Impact**: Production output contaminated; difficult to parse JSON output from simulations
- **Fix Required**: Remove all debug prints or replace with proper logging module

### 4. **Hard-coded Magic Numbers Throughout**
- **Location**: Multiple files
- **Examples**:
  - `run/runner.py`: Lines 111-113 (model hidden dims `[200, 1000, 1000, 200]`)
  - `src/losses.py`: Line 131 (`-10*dt**2` hardcoded loss regularization)
  - `src/losses.py`: Lines 205, 321 (eps values `1e-12`)
- **Severity**: MEDIUM
- **Impact**: Model architecture and hyperparameters scattered across codebase; difficult to refactor
- **Fix Required**: Centralize in config or model factory functions

### 5. **Optional Dependencies Not Properly Handled**
- **Location**: `run/runner.py`, lines 242-246
- **Issue**: W&B import happens at runtime with generic error handling:
  ```python
  try:
      import wandb as wandb_lib
  except ImportError as exc:
      raise RuntimeError("wandb is not installed...") from exc
  ```
- **Severity**: MEDIUM
- **Impact**: Runtime failures rather than clear setup instructions; no version pinning
- **Fix Required**: Add to requirements.txt with optional extras; document setup instructions in USAGE.md

### 6. **Config Serialization Issues**
- **Location**: `src/config.py`, lines 211-212
- **Issue**: Type conversion assumes `external_field_position` is always list when converting from dict:
  ```python
  if "external_field_position" in kwargs and isinstance(kwargs["external_field_position"], list):
      kwargs["external_field_position"] = tuple(kwargs["external_field_position"])
  ```
- **Severity**: MEDIUM
- **Impact**: Silent data loss if config serialization format changes; fragile type handling
- **Fix Required**: Add comprehensive type validation and conversion

---

## Medium Priority Issues

### 7. **Loss Function Design Ambiguity**
- **Location**: `src/losses.py`, `src/losses_history.py`
- **Issue**: Multiple loss calculation strategies without clear documentation:
  - `loss_energy_mean`, `loss_energy_last`, `loss_energy_next`, `loss_energy_max` all computed but combined with simple averaging
  - No principled weighting scheme
  - Energy loss bounds and angular momentum loss bounds set independently with unclear interaction
- **Severity**: MEDIUM
- **Impact**: Model behavior not reproducible; hyperparameter tuning difficult
- **Fix Required**: Document loss function design decisions; add ablation study

### 8. **Batch Dimension Handling Inconsistency**
- **Location**: `src/losses.py`, `src/losses_history.py`, `src/history_buffer.py`
- **Issue**: Multiple `.dim()` checks and `.unsqueeze()` operations suggest fragile shape handling:
  ```python
  if batch.dim() == 1:
      batch = batch.unsqueeze(0)  # (1, 2)
  if params.dim() == 1:
      params = params.unsqueeze(0)
  ```
- **Severity**: MEDIUM
- **Impact**: Shape-related bugs may only appear with specific batch sizes or input configurations
- **Fix Required**: Add comprehensive shape validation; create batch normalization helper

### 9. **History Buffer Edge Cases**
- **Location**: `src/history_buffer.py`, lines 254-292
- **Issue**: Padding logic repeats oldest state if history is incomplete:
  ```python
  if len(past_list) < self.history_len:
      pad_count = self.history_len - len(past_list)
      if past_list:
          past_list = [past_list[0]] * pad_count + past_list
  ```
- **Severity**: MEDIUM
- **Impact**: First steps of training use artificial padding rather than real history; may distort learning
- **Fix Required**: Document behavior clearly; consider alternative initialization strategies

### 10. **Missing Validation in Model Adapter**
- **Location**: `src/model_adapter.py`, lines 70-95
- **Issue**: `predict_dt()` method has minimal error handling:
  - No validation of model output shape
  - Silent dimension squeezing that could mask issues
  - No check for NaN/Inf in predictions
- **Severity**: MEDIUM
- **Impact**: Invalid predictions silently propagate through simulation
- **Fix Required**: Add shape validation and numeric stability checks

---

## Low-Medium Priority Issues

### 11. **Incomplete Feature Type Implementation**
- **Location**: `src/model_adapter.py`, lines 33-36
- **Issue**: `feature_mode()` method silently falls back to "basic" for unrecognized types:
  ```python
  if self.config.feature_type in ("basic", "rich"):
      return self.config.feature_type
  return "basic"  # Silent fallback!
  ```
- **Severity**: MEDIUM-LOW
- **Impact**: Configuration errors silently ignored
- **Fix Required**: Raise ValueError for unknown feature types

### 12. **Legacy Code Accumulation**
- **Location**: Multiple locations
- **Files**:
  - `run/legacy/` directory exists but unclear status
  - `src/obsolete/losses.py` file present but unused
  - Deleted files in git: `ML_history_multi_wandb.py`, `ML_history_wandb.py`, `integration_sanity.py`, `tidal_sanity.py`
- **Severity**: LOW
- **Impact**: Code confusion; cluttered repository
- **Fix Required**: Document deprecation; clean up or archive legacy code

### 13. **NFS Temporary Files in Git**
- **Location**: `run/.nfs*` files
- **Issue**: Temporary NFS lock files being tracked
- **Severity**: LOW
- **Impact**: Repository bloat; potential merge conflicts
- **Fix Required**: Add to `.gitignore`

### 14. **Missing Test Coverage**
- **Location**: Repository root
- **Issue**: No `tests/` directory found; no test files in codebase
- **Severity**: MEDIUM-LOW
- **Impact**: No regression testing; difficult to validate changes
- **Fix Required**: Establish test suite for core physics calculations and model training

### 15. **Type Hints Inconsistency**
- **Location**: Throughout codebase
- **Issue**: Partial type hints; mix of PEP 484 and future annotations
  - `src/config.py`: Uses `|` union syntax with `from __future__ import annotations`
  - Many functions have partial type hints
- **Severity**: LOW
- **Impact**: IDE support reduced; harder to maintain
- **Fix Required**: Standardize on complete type hints; run mypy/pyright

---

## Robustness & Error Handling

### 16. **Insufficient Input Validation**
- **Locations**:
  - `run/runner.py`: `_load_ic()` doesn't validate file format
  - `run/runner.py`: `_make_sim_particles()` assumes consistent row format
  - `src/external_potentials.py`: Minimal dimension checking
- **Severity**: MEDIUM
- **Impact**: Cryptic errors with malformed inputs
- **Fix Required**: Add comprehensive input validation with clear error messages

### 17. **Silent Numerical Issues**
- **Location**: `src/losses.py` and `src/losses_history.py`
- **Issue**: NaN/Inf handling with blanket replacements:
  ```python
  rel_dE = torch.where(
      torch.isfinite(rel_dE),
      rel_dE,
      torch.full_like(rel_dE, 1.0)  # Replace inf/nan with 1.0
  )
  ```
- **Severity**: MEDIUM-LOW
- **Impact**: Divergent simulations silently treated as valid; loss history obscured
- **Fix Required**: Log when this occurs; consider early stopping

### 18. **Device/Dtype Mismatch Vulnerabilities**
- **Location**: `run/runner.py`, `src/model_adapter.py`
- **Issue**: Multiple `.to(device=..., dtype=...)` calls; no centralized consistency checking
- **Severity**: MEDIUM-LOW
- **Impact**: Silent dtype conversion losses; mixed precision issues
- **Fix Required**: Create device/dtype context manager

---

## Performance & Scaling Concerns

### 19. **No Batch Size Flexibility in Simulation**
- **Location**: `run/runner.py`, `simulators/nbody_simulator.py`
- **Issue**: Simulation loop processes single particles sequentially; doesn't batch
- **Severity**: MEDIUM-LOW
- **Impact**: Poor GPU utilization for ensemble simulations
- **Fix Required**: Implement parallel ensemble simulation

### 20. **Checkpoint Duplication & Bloat**
- **Location**: `src/checkpoint.py`, lines 52-62
- **Issue**: Saves both `model_state_dict` and `model_state` (identical), and both optimizer keys:
  ```python
  "model_state_dict": model_state,
  "model_state": model_state,  # Duplicate!
  "optimizer_state_dict": optimizer_state,
  "optimizer_state": optimizer_state,  # Duplicate!
  ```
- **Severity**: LOW
- **Impact**: 2x checkpoint file size
- **Fix Required**: Clean up; keep only one version with proper naming

---

## Documentation & Maintainability

### 21. **Scattered Configuration Parameters**
- **Location**: Across multiple files
- **Issue**: Configuration parameters in:
  - `src/config.py`: CLI argument definitions
  - `run/runner.py`: Hidden layer dimensions, eps values
  - `src/losses.py`: Loss scaling, epsilon values
  - `src/history_buffer.py`: Feature extraction logic
- **Severity**: MEDIUM
- **Impact**: Difficult to understand and modify system behavior
- **Fix Required**: Centralize configuration in Config class

### 22. **Insufficient Function Documentation**
- **Location**: Core functions throughout
- **Examples**:
  - `src/losses_history.py`: Long functions with complex logic but minimal docstrings
  - `src/nbody_features.py`: Feature computation logic undocumented
  - `src/history_buffer.py`: Padding logic not explained
- **Severity**: LOW-MEDIUM
- **Impact**: Onboarding difficult; maintenance errors likely
- **Fix Required**: Add comprehensive docstrings with examples

### 23. **Git Repository State**
- **Location**: Repository root
- **Issue**: Multiple uncommitted deletions and new files
  - 6 deleted files (training scripts)
  - 2 new documentation files (CLAUDE.md, USAGE.md)
  - NFS lock files
- **Severity**: LOW
- **Impact**: Unclear project state; potential for lost work
- **Fix Required**: Clean up and commit with clear messages

---

## Specific Code Smells

### 24. **Redundant Calculations in History Buffer**
- **Location**: `src/losses_history.py`, lines 171-197
- **Issue**: Angular momentum computed twice with slightly different methods:
  - Lines 62-92: `_angular_momentum_mag()` computes per-step magnitude
  - Lines 145-170: `_compute_angular_momentum_batch()` duplicates logic
- **Severity**: LOW
- **Impact**: Maintenance burden; potential for divergent calculations
- **Fix Required**: Extract to single utility function

### 25. **Unused Imports and Dead Code**
- **Location**: `src/structures.py`, lines 1-29
- **Issue**: `SimpleNN` class defined but never used; `_initialize_weights()` method not called
- **Severity**: LOW
- **Impact**: Code confusion
- **Fix Required**: Remove unused code or document usage

### 26. **Long Parameter Lists**
- **Location**: Multiple loss functions
- **Examples**:
  - `loss_fn_batch_history_batch()`: 12 parameters
  - `loss_fn_batch_history()`: 11 parameters
- **Severity**: LOW
- **Impact**: Difficult to call; error-prone
- **Fix Required**: Create dataclass for loss hyperparameters

---

## Security & Data Integrity

### 27. **No Input Bounds Checking**
- **Location**: `run/runner.py`, simulation setup
- **Issue**: No validation that step counts, durations, particle counts are reasonable
- **Severity**: LOW
- **Impact**: Potential for runaway resource consumption
- **Fix Required**: Add reasonable bounds and warnings

### 28. **Pickle-based Checkpoint Format**
- **Location**: `src/checkpoint.py`
- **Issue**: Uses `torch.save()` which uses pickle; vulnerable to arbitrary code execution
- **Severity**: LOW (when loading trusted checkpoints only)
- **Impact**: Unsafe to load checkpoints from untrusted sources
- **Fix Required**: Consider safetensors or explicit format specification

---

## Recommendations Priority Matrix

| Priority | Issue | Effort | Impact |
|----------|-------|--------|--------|
| CRITICAL | Return value errors (1-2) | Trivial | Blocking |
| HIGH | Debug prints (3) | Low | Cosmetic but disruptive |
| HIGH | Magic numbers (4) | Medium | Maintainability |
| HIGH | Optional deps (5) | Low | User experience |
| MEDIUM | Loss function design (7) | High | Scientific validity |
| MEDIUM | Batch handling (8) | Medium | Robustness |
| MEDIUM | Config consistency (21) | Medium | Maintainability |
| MEDIUM | Test coverage (14) | High | Reliability |
| LOW | Legacy cleanup (12) | Low | Code hygiene |
| LOW | Checkpoint bloat (20) | Trivial | Performance |

---

## Action Items

### Phase 1: Critical Fixes (Required before next release)
- [ ] Fix return value bugs in losses.py and losses_history.py
- [ ] Remove all debug print statements
- [ ] Document current training configuration

### Phase 2: Robustness (Next sprint)
- [ ] Add comprehensive input validation
- [ ] Create basic test suite for core physics
- [ ] Standardize batch handling with helper functions
- [ ] Add proper logging module

### Phase 3: Maintainability (Ongoing)
- [ ] Consolidate magic numbers into Config
- [ ] Document all loss function design decisions
- [ ] Add type hints to all functions
- [ ] Clean up legacy code

### Phase 4: Long-term Improvements (Future)
- [ ] Implement parallel ensemble simulation
- [ ] Create performance profiling benchmarks
- [ ] Consider refactoring to use dataclass for loss params
- [ ] Migrate from pickle to safetensors for checkpoints
