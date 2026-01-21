# STATE: AITimeStepper Training Refactor

**Initialized:** 2026-01-20
**Current Phase:** 2 (completed)
**Overall Progress:** 2/7 phases complete

---

## Phase Status

| Phase | Status | Plans Complete | Notes |
|-------|--------|----------------|-------|
| 1. Configuration Parameters | **DONE** | 2/2 | Added steps_per_epoch, validation for training params |
| 2. History Buffer Zero-Padding | **DONE** | 2/2 | Zero-padding in all feature extraction methods |
| 3. Part 1: Trajectory Collection | Pending | 0/5 | Implement accept/reject loop |
| 4. Part 2: Generalization Training | Pending | 0/4 | Train on minibatches until convergence |
| 5. Unified Epoch Structure | Pending | 0/4 | Combine Part 1 + Part 2 |
| 6. Integration into runner.py | Pending | 0/5 | Replace existing run_training() |
| 7. Cleanup Legacy Code | Pending | 0/2 | Remove old training loop |

---

## Current Work

**Phase:** 2 (completed)
**Plan:** All plans executed
**Status:** Ready for Phase 3

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

### Key Files
- `/u/gkerex/projects/AITimeStepper/src/config.py` - Config dataclass
- `/u/gkerex/projects/AITimeStepper/src/history_buffer.py` - HistoryBuffer class (zero-padding implemented)
- `/u/gkerex/projects/AITimeStepper/run/runner.py` - run_training() function
- `/u/gkerex/projects/AITimeStepper/src/losses.py` - Loss functions (analytic)
- `/u/gkerex/projects/AITimeStepper/src/losses_history.py` - Loss functions (history)

### Next Steps
1. Phase 3: Implement Part 1 trajectory collection loop
   - TRAIN-01: Predict dt, integrate one step, check energy
   - TRAIN-02: Reject and retrain if energy exceeds threshold
   - TRAIN-03: Accept and record if energy within threshold
   - TRAIN-04: Collect N steps per epoch
   - HIST-02: Discard warmup steps once real trajectory exists
2. Then Phase 4: Implement Part 2 generalization training

---

## Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-01-20 | 7 phases for standard depth | Requirements naturally group into config, history, two training parts, integration, cleanup |
| 2026-01-20 | Zero-padding for history bootstrap | Cleaner than oldest-state repeat, warmup discarded anyway |
| 2026-01-20 | No retry cap in Part 1 | User preference: iterate until physics satisfied |
| 2026-01-20 | Single model for both parts | Design decision from requirements |
| 2026-01-20 | HIST-02 moved to Phase 3 | Warmup discard is training loop concern, not buffer concern |

---

*State initialized: 2026-01-20*
*Last updated: 2026-01-20 (Phase 2 complete)*
