# STATE: AITimeStepper Training Refactor

**Initialized:** 2026-01-20
**Current Phase:** 2 (in progress)
**Overall Progress:** 1/7 phases complete (Phase 2: 2/3 plans)

---

## Phase Status

| Phase | Status | Plans Complete | Notes |
|-------|--------|----------------|-------|
| 1. Configuration Parameters | **DONE** | 2/2 | Added steps_per_epoch, validation for training params |
| 2. History Buffer Zero-Padding | **IN PROGRESS** | 2/3 | Plans 02-01, 02-02 complete: zero-padding + extended tests |
| 3. Part 1: Trajectory Collection | Pending | 0/5 | Implement accept/reject loop |
| 4. Part 2: Generalization Training | Pending | 0/4 | Train on minibatches until convergence |
| 5. Unified Epoch Structure | Pending | 0/4 | Combine Part 1 + Part 2 |
| 6. Integration into runner.py | Pending | 0/5 | Replace existing run_training() |
| 7. Cleanup Legacy Code | Pending | 0/2 | Remove old training loop |

---

## Current Work

**Phase:** 2 (History Buffer Zero-Padding)
**Plan:** 02-01 and 02-02 completed, 02-03 pending
**Status:** Zero-padding and testing complete, need to update model adapter and integration tests

---

## Completed Work

### Phase 1: Configuration Parameters (2026-01-20)
- **PLAN-01**: Added `steps_per_epoch: int = 1` field and `--steps-per-epoch` CLI argument
- **PLAN-02**: Added validation for `epochs >= 1`, `steps_per_epoch >= 1`, `energy_threshold > 0`
- **Commit**: 33b4a9f
- **Research finding**: `energy_threshold` already existed (default 2e-4), only `steps_per_epoch` needed adding

### Phase 2: History Buffer Zero-Padding (2026-01-20)
- **PLAN-02-01**: Zero-state factory and zero-padding implementation
  - **Commit 286a5cc**: Added `_zero_state()` static method to HistoryBuffer class
  - **Commit f168818**: Implemented zero-padding in features_for(), features_for_batch(), and features_for_histories()
  - **Commit d7f744d**: Added `_test_zero_padding()` unit test function
  - **Impact**: Cleaner signal to model during bootstrap (zeros indicate "no data" vs false repetition)
- **PLAN-02-02**: Extended test coverage for batch methods
  - **Commit eef826b**: Extended `_test_zero_padding()` with 3 new test cases
  - **Tests added**: features_for_batch(), features_for_histories(), delta_mag feature type
  - **Discovery**: Wave 1 executor already implemented zero-padding in all three methods during Plan 02-01
  - **Impact**: Comprehensive test coverage for all feature extraction methods and feature types

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
- History buffer currently pads with oldest state (needs change)
- ModelAdapter abstracts feature extraction (analytic vs history)

### Key Files
- `/u/gkerex/projects/AITimeStepper/src/config.py` - Config dataclass
- `/u/gkerex/projects/AITimeStepper/src/history_buffer.py` - HistoryBuffer class
- `/u/gkerex/projects/AITimeStepper/run/runner.py` - run_training() function
- `/u/gkerex/projects/AITimeStepper/src/losses.py` - Loss functions (analytic)
- `/u/gkerex/projects/AITimeStepper/src/losses_history.py` - Loss functions (history)

### Next Steps
1. Complete Phase 2 remaining plan (02-03: model adapter and integration tests)
2. Then Phase 3: Implement trajectory collection loop
3. Then Phase 4: Implement generalization training

---

## Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-01-20 | 7 phases for standard depth | Requirements naturally group into config, history, two training parts, integration, cleanup |
| 2026-01-20 | Zero-padding for history bootstrap | Cleaner than oldest-state repeat, warmup discarded anyway |
| 2026-01-20 | No retry cap in Part 1 | User preference: iterate until physics satisfied |
| 2026-01-20 | Single model for both parts | Design decision from requirements |

---

*State initialized: 2026-01-20*
*Last updated: 2026-01-20 (Phase 2 Plan 02-02 complete)*
