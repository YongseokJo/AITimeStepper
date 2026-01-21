# STATE: AITimeStepper Training Refactor

**Initialized:** 2026-01-20
**Current Phase:** 4 (in progress)
**Overall Progress:** 3/7 phases complete (Plan 04-02 done)

---

## Phase Status

| Phase | Status | Plans Complete | Notes |
|-------|--------|----------------|-------|
| 1. Configuration Parameters | **DONE** | 2/2 | Added steps_per_epoch, validation for training params |
| 2. History Buffer Zero-Padding | **DONE** | 2/2 | Zero-padding in all feature extraction methods |
| 3. Part 1: Trajectory Collection | **DONE** | 4/4 | All primitives, retrain loop, orchestrator, and tests complete |
| 4. Part 2: Generalization Training | **IN PROGRESS** | 2/4 | Plan 01-02 complete: core functions and tests |
| 5. Unified Epoch Structure | Pending | 0/4 | Combine Part 1 + Part 2 |
| 6. Integration into runner.py | Pending | 0/5 | Replace existing run_training() |
| 7. Cleanup Legacy Code | Pending | 0/2 | Remove old training loop |

---

## Current Work

**Phase:** 4 (in progress)
**Plan:** 04-02 (complete)
**Status:** Unit tests for generalization training complete

### Plan 04-02 Complete
- Created `tests/test_generalization_training.py` (402 lines, 16 test cases)
- TestSampleMinibatch: 4 tests for minibatch sampling behavior
- TestEvaluateMinibatch: 4 tests for evaluation and particle immutability
- TestGeneralizeOnTrajectory: 8 tests for convergence, edge cases, training behavior
- All 16 tests pass in under 6 seconds
- Commits: 5f8d2be, 1c086d9, add0fd4

### Plan 04-01 Complete
- Created `src/generalization_training.py` (241 lines)
- `sample_minibatch()`: Random sampling from trajectory list
- `evaluate_minibatch()`: Single-step evaluation with energy threshold
- `generalize_on_trajectory()`: Convergence loop until all pass
- Edge cases handled: empty trajectory, small trajectory
- All functions exported from `src/__init__.py`
- Commits: d84477b, 80b86c0

---

## Completed Work

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
- `/u/gkerex/projects/AITimeStepper/tests/test_trajectory_collection.py` - Unit tests (NEW in Plan 03-04)
- `/u/gkerex/projects/AITimeStepper/src/generalization_training.py` - Generalization training module (NEW in Plan 04-01)
- `/u/gkerex/projects/AITimeStepper/tests/test_generalization_training.py` - Unit tests (NEW in Plan 04-02)
- `/u/gkerex/projects/AITimeStepper/run/runner.py` - run_training() function
- `/u/gkerex/projects/AITimeStepper/src/losses.py` - Loss functions (analytic)
- `/u/gkerex/projects/AITimeStepper/src/losses_history.py` - Loss functions (history)

### Next Steps
1. Phase 4, Plan 04-03: Integration tests combining Part 1 + Part 2
2. Phase 4, Plan 04-04: Documentation and examples
3. Phase 5: Unified epoch structure combining trajectory collection + generalization

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
*Last updated: 2026-01-21 (Phase 4, Plan 02 complete)*
