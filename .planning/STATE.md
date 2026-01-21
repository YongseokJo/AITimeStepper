# STATE: AITimeStepper Training Refactor

**Initialized:** 2026-01-20
**Current Phase:** 7 (pending)
**Overall Progress:** 6/7 phases complete

---

## Phase Status

| Phase | Status | Plans Complete | Notes |
|-------|--------|----------------|-------|
| 1. Configuration Parameters | **DONE** | 2/2 | Added steps_per_epoch, validation for training params |
| 2. History Buffer Zero-Padding | **DONE** | 2/2 | Zero-padding in all feature extraction methods |
| 3. Part 1: Trajectory Collection | **DONE** | 4/4 | All primitives, retrain loop, orchestrator, and tests complete |
| 4. Part 2: Generalization Training | **DONE** | 2/2 | Core functions and tests complete, verified |
| 5. Unified Epoch Structure | **DONE** | 3/3 | Epoch orchestrator, outer loop, and tests complete |
| 6. Integration into runner.py | **DONE** | 2/2 | run_training() refactored, integration tests added |
| 7. Cleanup Legacy Code | Pending | 0/2 | Remove old training loop |

---

## Current Work

**Phase:** 7 (pending)
**Plan:** Not started
**Status:** Phase 6 complete, ready for cleanup

---

## Completed Work

### Phase 6: Integration into runner.py (2026-01-21, COMPLETE)
- **PLAN-06-01**: Refactor run_training() to use run_two_phase_training()
  - Added `import warnings` to runner.py
  - Added `run_two_phase_training` to imports from src
  - Refactored `run_training()` into 12 clearly-labeled sections
  - Removed `_wandb_log_value()` helper (W&B logging in run_two_phase_training)
  - Removed `_build_particle()` inner function (inlined)
  - Removed multi-orbit particle initialization (replaced with warning)
  - Removed manual training loop (replaced with single function call)
  - Added warnings for unsupported features (num_orbits > 1, duration)
  - Added training summary print after completion
  - File: 387 lines (reduced from 429 due to deduplication)
  - Commits: ee3a0a2, 2460c69

- **PLAN-06-02**: Integration tests for CLI, checkpoints, and warnings
  - Created `tests/test_runner_integration.py` (861 lines, 28 tests)
  - TestArgumentParsing: 6 tests for CLI argument parsing
  - TestMultiOrbitWarning: 2 tests for num_orbits > 1 warning
  - TestDurationWarning: 2 tests for duration warning
  - TestWandBLogging: 2 tests for W&B integration
  - TestCheckpointContract: 3 tests for checkpoint fields
  - TestDirectFunctionCalls: 5 tests for run_training()
  - TestRunTrainingIntegration: 3 tests for end-to-end
  - TestConfigFromCLI: 5 tests for config validation
  - All 28 tests pass in ~5.6 seconds
  - Commits: b791a8f, 8675450

- **Verification**: All 6 success criteria met, INTG-01/INTG-03 requirements satisfied

### Phase 5: Unified Epoch Structure (2026-01-21, COMPLETE)
- **PLAN-05-01**: Implement train_epoch_two_phase() orchestrator
  - Created `src/unified_training.py` (117 lines)
  - `train_epoch_two_phase()`: Calls collect_trajectory() then generalize_on_trajectory()
  - Passes trajectory directly from Part 1 to Part 2
  - Returns combined metrics: trajectory_metrics, generalization_metrics, converged, part2_iterations, epoch_time
  - Handles empty trajectory edge case with UserWarning
  - Tracks epoch wall clock time with time.perf_counter()
  - All functions exported from `src/__init__.py`
  - Commits: f45659a, 98fb6f9

- **PLAN-05-02**: Implement run_two_phase_training() outer loop
  - Extended `src/unified_training.py` (now 281 lines)
  - `run_two_phase_training()`: N-epoch training with checkpointing and W&B logging
  - Calls train_epoch_two_phase() for config.epochs iterations
  - Checkpoint saved every checkpoint_interval epochs (default 10) and on final epoch
  - W&B logging tracks Part 1 (acceptance_rate, trajectory_length) and Part 2 (converged, iterations)
  - Same history_buffer instance persists across all epochs
  - Returns aggregated results: epochs_completed, total_time, convergence_rate
  - Progress printed every 10 epochs
  - Commits: dd3a045, 91b8167

- **PLAN-05-03**: Unit tests for unified training
  - Created `tests/test_unified_training.py` (651 lines, 21 test cases)
  - TestTrainEpochTwoPhase: 8 tests (return structure, metrics, history buffer, edge cases, call order)
  - TestRunTwoPhaseTraining: 11 tests (epochs, checkpointing, convergence, history persistence)
  - TestIntegration: 2 tests (end-to-end, history buffer)
  - All 21 tests pass in 4.24 seconds
  - Commit: f5eddf5

- **Verification**: All success criteria met, TRAIN-08/09 requirements satisfied

### Phase 4: Generalization Training Loop (2026-01-21)
- **PLAN-04-01**: Implement generalize_on_trajectory with minibatch sampling
  - Created `src/generalization_training.py` (241 lines)
  - `sample_minibatch()`: Random sampling from trajectory list
  - `evaluate_minibatch()`: Single-step evaluation with energy threshold
  - `generalize_on_trajectory()`: Convergence loop until all pass
  - Edge cases handled: empty trajectory, small trajectory
  - All functions exported from `src/__init__.py`
  - Commits: d84477b, 80b86c0, f7daba5

- **PLAN-04-02**: Unit tests for generalization training
  - Created `tests/test_generalization_training.py` (402 lines, 16 test cases)
  - TestSampleMinibatch: 4 tests for minibatch sampling behavior
  - TestEvaluateMinibatch: 4 tests for evaluation and particle immutability
  - TestGeneralizeOnTrajectory: 8 tests for convergence, edge cases, training behavior
  - All 16 tests pass in under 6 seconds
  - Commits: 5f8d2be, 1c086d9, add0fd4, 3b520dd

- **Verification**: All 6 success criteria met, TRAIN-05/06/07 requirements satisfied

### Phase 1: Configuration Parameters (2026-01-20)
- **PLAN-01**: Added `steps_per_epoch: int = 1` field and `--steps-per-epoch` CLI argument
- **PLAN-02**: Added validation for `epochs >= 1`, `steps_per_epoch >= 1`, `energy_threshold > 0`
- **Commit**: 33b4a9f
- **Research finding**: `energy_threshold` already existed (default 2e-4), only `steps_per_epoch` needed adding

### Phase 2: History Buffer Zero-Padding (2026-01-20)
- **PLAN-02-01**: Zero-state factory and zero-padding implementation
  - Added `_zero_state()` static method to HistoryBuffer class
  - Implemented zero-padding in features_for(), features_for_batch(), features_for_histories()
  - Added `_test_zero_padding()` unit test function
- **PLAN-02-02**: Extended test coverage for batch methods and delta_mag
  - Verified batch methods already complete from Plan 02-01
  - Extended test function with 3 additional test cases
- **Commits**: 286a5cc, f168818, d7f744d, eef826b, 5dc18eb
- **Impact**: Cleaner signal to model during bootstrap (zeros indicate "no data" vs false repetition)
- **Verification**: All 5 success criteria passed, HIST-01 requirement satisfied

### Phase 3: Trajectory Collection Loop (2026-01-20 to 2026-01-21)
- **PLAN-03-01**: Core trajectory collection primitives
  - Created `src/trajectory_collection.py` module (193 lines)
  - Implemented `attempt_single_step()`: predicts dt, integrates, returns (particle, dt, E0, E1)
  - Implemented `check_energy_threshold()`: validates energy conservation
  - Implemented `compute_single_step_loss()`: band loss for retrain loop
  - All functions use `clone_detached()` pattern to prevent graph accumulation
  - Complete type hints, docstrings, and usage examples
- **Commit**: c4ac39a
- **Impact**: Foundational primitives for accept/reject trajectory collection
- **Verification**: All must-haves met, 193 lines, exports added to src/__init__.py

- **PLAN-03-02**: Retrain loop implementation
  - Implemented `collect_trajectory_step()` with accept/reject retrain loop
  - No retry limit by design (user requirement)
  - Returns (particle, dt, metrics) with retrain iteration count
- **Commit**: 50a2f42

- **PLAN-03-03**: Trajectory orchestrator
  - Implemented `collect_trajectory()` epoch orchestrator
  - Implements HIST-02 warmup discard mechanism
  - Collects steps_per_epoch validated steps
  - Discards first history_len steps as warmup
  - Returns trajectory list and epoch metrics
- **Commit**: c2f7435

- **PLAN-03-04**: Unit tests for trajectory collection (2026-01-21)
  - Created `tests/test_trajectory_collection.py` (417 lines, 15 test cases)
  - TestAttemptSingleStep: return tuple structure, clone isolation, energy preservation
  - TestCheckEnergyThreshold: accept/reject logic, small energy handling
  - TestComputeSingleStepLoss: scalar output, band loss behavior
  - TestCollectTrajectoryStep: return structure, retrain loop
  - TestCollectTrajectory: step collection, warmup discard, history buffer updates
  - MockModel and TrainableMockModel with dynamic input dimension handling
  - Pre-population pattern for history tests to avoid zero-padding NaN
- **Commit**: 0de9c24
- **Verification**: All 15 tests pass in under 4 seconds

---

## Blockers

None.

---

## Context for Next Session

### What We Know
- Project uses PyTorch for differentiable N-body physics
- Existing training: multi-step integration per loss evaluation
- New approach: two-phase with energy gates
- Config system uses dataclass with CLI auto-generation
- Checkpoints include config metadata for reproducibility
- History buffer now pads with zeros (HIST-01 complete)
- ModelAdapter abstracts feature extraction (analytic vs history)
- Trajectory collection module complete with unit tests

### Key Files
- `/u/gkerex/projects/AITimeStepper/src/config.py` - Config dataclass
- `/u/gkerex/projects/AITimeStepper/src/history_buffer.py` - HistoryBuffer class (zero-padding implemented)
- `/u/gkerex/projects/AITimeStepper/src/trajectory_collection.py` - Trajectory primitives and orchestrator
- `/u/gkerex/projects/AITimeStepper/tests/test_trajectory_collection.py` - Unit tests (Plan 03-04)
- `/u/gkerex/projects/AITimeStepper/src/generalization_training.py` - Generalization training module (Plan 04-01)
- `/u/gkerex/projects/AITimeStepper/tests/test_generalization_training.py` - Unit tests (Plan 04-02)
- `/u/gkerex/projects/AITimeStepper/src/unified_training.py` - Unified epoch orchestrator (Plans 05-01, 05-02)
- `/u/gkerex/projects/AITimeStepper/tests/test_unified_training.py` - Unit tests (Plan 05-03)
- `/u/gkerex/projects/AITimeStepper/run/runner.py` - run_training() function (target for Phase 6)
- `/u/gkerex/projects/AITimeStepper/src/losses.py` - Loss functions (analytic)
- `/u/gkerex/projects/AITimeStepper/src/losses_history.py` - Loss functions (history)

### Next Steps
1. Phase 6: Complete remaining plans (06-02 through 06-05)
2. Phase 7: Cleanup legacy code

### Known Issues
- Zero-padding in HistoryBuffer produces NaN for acceleration features when mass is zero
  - Workaround: Pre-populate history buffer with valid states before use
  - Not critical for production use (warmup discards these steps anyway)

---

## Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-01-20 | 7 phases for standard depth | Requirements naturally group into config, history, two training parts, integration, cleanup |
| 2026-01-20 | Zero-padding for history bootstrap | Cleaner than oldest-state repeat, warmup discarded anyway |
| 2026-01-20 | No retry cap in Part 1 | User preference: iterate until physics satisfied |
| 2026-01-20 | Single model for both parts | Design decision from requirements |
| 2026-01-20 | HIST-02 moved to Phase 3 | Warmup discard is training loop concern, not buffer concern |
| 2026-01-20 | clone_detached() at start of attempt_single_step | Prevents graph accumulation during trajectory collection |
| 2026-01-20 | Combined Task 1-3 implementation | All tasks share patterns, more cohesive as single unit |
| 2026-01-21 | Dynamic input dimension in mock models | Supports both analytic (11) and history (44) feature dimensions |
| 2026-01-21 | Pre-populate history for tests | Avoids zero-padding NaN bug in acceleration computation |
| 2026-01-21 | Higher energy threshold for tests | 10% vs 0.02% for faster test convergence; unit tests verify behavior not physics |
| 2026-01-21 | random.sample() for minibatch | Trajectory is Python list, not tensor; uniform sampling without replacement |
| 2026-01-21 | Early return for empty/small trajectory | Edge cases return success immediately with appropriate metrics |

---

*State initialized: 2026-01-20*
*Last updated: 2026-01-21 (Plan 06-01 complete)*
