# STATE: AITimeStepper Training Refactor

**Initialized:** 2026-01-20
**Current Phase:** 3 (in progress)
**Overall Progress:** 2/7 phases complete, 1/5 plans in Phase 3

---

## Phase Status

| Phase | Status | Plans Complete | Notes |
|-------|--------|----------------|-------|
| 1. Configuration Parameters | **DONE** | 2/2 | Added steps_per_epoch, validation for training params |
| 2. History Buffer Zero-Padding | **DONE** | 2/2 | Zero-padding in all feature extraction methods |
| 3. Part 1: Trajectory Collection | **In Progress** | 1/5 | Plan 03-01 complete: trajectory primitives |
| 4. Part 2: Generalization Training | Pending | 0/4 | Train on minibatches until convergence |
| 5. Unified Epoch Structure | Pending | 0/4 | Combine Part 1 + Part 2 |
| 6. Integration into runner.py | Pending | 0/5 | Replace existing run_training() |
| 7. Cleanup Legacy Code | Pending | 0/2 | Remove old training loop |

---

## Current Work

**Phase:** 3 (in progress)
**Plan:** 03-01 (completed)
**Status:** Trajectory collection primitives complete, ready for Plan 03-02 (retrain loop)

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

### Phase 3: Trajectory Collection Loop (2026-01-20)
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
- `/u/gkerex/projects/AITimeStepper/src/trajectory_collection.py` - Trajectory primitives (NEW in Plan 03-01)
- `/u/gkerex/projects/AITimeStepper/run/runner.py` - run_training() function
- `/u/gkerex/projects/AITimeStepper/src/losses.py` - Loss functions (analytic)
- `/u/gkerex/projects/AITimeStepper/src/losses_history.py` - Loss functions (history)

### Next Steps
1. Phase 3: Continue trajectory collection loop implementation
   - ✅ Plan 03-01: Core trajectory primitives (complete)
   - Plan 03-02: Retrain loop (single-step gradient descent until accept)
   - Plan 03-03: Trajectory collector (main loop: attempt → reject/retrain → accept → record)
   - Plan 03-04: Epoch loop (collect N steps, discard warmup, build trajectory)
   - Plan 03-05: History buffer integration (update history after accept)
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
| 2026-01-20 | clone_detached() at start of attempt_single_step | Prevents graph accumulation during trajectory collection |
| 2026-01-20 | Combined Task 1-3 implementation | All tasks share patterns, more cohesive as single unit |

---

*State initialized: 2026-01-20*
*Last updated: 2026-01-20 (Phase 3, Plan 03-01 complete)*
