# STATE: AITimeStepper Training Refactor

**Initialized:** 2026-01-20
**Current Phase:** 1 (completed)
**Overall Progress:** 1/7 phases complete

---

## Phase Status

| Phase | Status | Plans Complete | Notes |
|-------|--------|----------------|-------|
| 1. Configuration Parameters | **DONE** | 2/2 | Added steps_per_epoch, validation for training params |
| 2. History Buffer Zero-Padding | Pending | 0/3 | Replace oldest-state padding with zero-padding |
| 3. Part 1: Trajectory Collection | Pending | 0/5 | Implement accept/reject loop |
| 4. Part 2: Generalization Training | Pending | 0/4 | Train on minibatches until convergence |
| 5. Unified Epoch Structure | Pending | 0/4 | Combine Part 1 + Part 2 |
| 6. Integration into runner.py | Pending | 0/5 | Replace existing run_training() |
| 7. Cleanup Legacy Code | Pending | 0/2 | Remove old training loop |

---

## Current Work

**Phase:** 1 (completed)
**Plan:** All plans executed
**Status:** Ready for Phase 2

---

## Completed Work

### Phase 1: Configuration Parameters (2026-01-20)
- **PLAN-01**: Added `steps_per_epoch: int = 1` field and `--steps-per-epoch` CLI argument
- **PLAN-02**: Added validation for `epochs >= 1`, `steps_per_epoch >= 1`, `energy_threshold > 0`
- **Commit**: 33b4a9f
- **Research finding**: `energy_threshold` already existed (default 2e-4), only `steps_per_epoch` needed adding

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
1. Start with Phase 1: Add config parameters
2. Then Phase 2: Modify history buffer padding
3. Then Phase 3: Implement trajectory collection loop

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
*Last updated: 2026-01-20 (initial creation)*
